/*
 * Harness for the Win64 unwinding tests (win64-tailcc-unwind-*.ll).
 *
 * Each test is an LLVM IR file that defines, besides the functions under test:
 *
 *   unwind_test_cases       {NULL,...}-terminated array of TestCase.
 *   unwind_checked_functions NULL-terminated array of the functions whose
 *                           instructions are checked (the functions under test
 *                           and the targets they tail call).
 *   unwind_all_functions    NULL-terminated array of every function in the
 *                           test, the last being a marker function defined
 *                           after all the others. Functions with no frame have
 *                           no unwind data, so they are found by address range.
 *   unwind_expected_context  CONTEXT captured by a runner just before it calls
 *                           the function under test.
 *   unwind_actual_context    CONTEXT captured just after the call returns.
 *
 * A runner is a C-ABI function that puts sentinel values in the nonvolatile
 * registers, captures them, calls the function under test and captures them
 * again. It returns 1 if the tail-call target saw the arguments it expected and
 * 0 otherwise. Because the runner calls a tail-call convention function, it is
 * compiled with a frame pointer.
 *
 * For every case and selector the harness
 *
 *   1. runs the runner normally and checks its result and that the nonvolatile
 *      registers are intact afterwards;
 *   2. runs it again with the trap flag set. At every instruction boundary
 *      inside a checked function, the single-step handler unwinds two frames
 *      with RtlVirtualUnwind: the first must give the runner with the
 *      nonvolatile registers of the runner's call, the second stepRun, which
 *      calls it. This is what a debugger or profiler would see;
 *   3. for each such boundary K, runs it again and lets the system exception
 *      dispatcher unwind from the K'th boundary to the __try below, which only
 *      works if the dispatcher finds the runner and this function.
 *
 * The runner's sentinel values are defined by Inputs/gen-win64-unwind-tests.py
 * and must match sentinelsMatch below. Two self-checks guard against a runner
 * that reused a sentinel register, which would make the comparisons above
 * meaningless: the captured context must hold the sentinels, and so must the
 * live registers when the function under test is entered.
 *
 * Progress is printed before each run, so if a broken unwind makes Windows kill
 * the process, the last line says where.
 */

#include <windows.h>
#include <intrin.h>
#include <stdio.h>
#include <string.h>

#pragma comment(lib, "kernel32.lib")

typedef struct {
  const char *Name;
  int (*Run)(int Selector);
  unsigned long long NumSelectors;
  /* Bit S is set if selector S ends in a tail call whose target checks its
     arguments. */
  unsigned long long TailMask;
  /* 1: the runner cannot keep sentinels in r13 and r14 (Swift's self and async
     context registers), so they are not compared. */
  unsigned long long Flags;
} TestCase;

extern const TestCase unwind_test_cases[];
extern void *const unwind_checked_functions[];
extern void *const unwind_all_functions[];
extern CONTEXT unwind_expected_context;
extern CONTEXT unwind_actual_context;

#define TRAP_FLAG 0x100u

static const TestCase *CurrentCase;
static int CurrentSelector;
static int CheckState;       /* check the unwound state at each step */
static LONG TargetStep;      /* let the dispatcher unwind here, if > 0 */
static volatile LONG Steps;  /* steps seen in checked functions */
static volatile LONG Reached;
static volatile int Stepping;  /* the runner is being single-stepped */
static int Failures;

static int isChecked(DWORD64 Begin) {
  for (void *const *F = unwind_checked_functions; *F; ++F)
    if ((DWORD64)*F == Begin)
      return 1;
  return 0;
}

/* The start addresses of all the functions, in increasing order. A function
   extends to the start of the next one. */
#define MAX_FUNCTIONS 256
static DWORD64 Starts[MAX_FUNCTIONS];
static int NumStarts;

static void sortFunctions(void) {
  for (void *const *F = unwind_all_functions; *F; ++F) {
    if (NumStarts == MAX_FUNCTIONS) {
      printf("FAIL: too many functions in the test\n");
      ++Failures;
      return;
    }
    DWORD64 Start = (DWORD64)*F;
    int I = NumStarts++;
    for (; I > 0 && Starts[I - 1] > Start; --I)
      Starts[I] = Starts[I - 1];
    Starts[I] = Start;
  }
}

/* Whether Pc is in a function under test or a function it tail calls. This
   does not use the unwind data: a function with no frame has none. */
static int isCheckedPc(DWORD64 Pc) {
  int I = NumStarts;
  while (I > 0 && Starts[I - 1] > Pc)
    --I;
  /* I is the number of functions that start at or before Pc. The last function
     is the marker, which ends nothing, so Pc must be before it. */
  return I > 0 && I < NumStarts && isChecked(Starts[I - 1]);
}

static int sameRegisters(const CONTEXT *A, const CONTEXT *B,
                         unsigned long long Flags) {
  if (A->Rbx != B->Rbx || A->Rbp != B->Rbp || A->Rsi != B->Rsi ||
      A->Rdi != B->Rdi || A->R12 != B->R12 || A->R15 != B->R15)
    return 0;
  if (!(Flags & 1) && (A->R13 != B->R13 || A->R14 != B->R14))
    return 0;
  const M128A *X = &A->Xmm6, *Y = &B->Xmm6;
  for (int I = 0; I != 10; ++I)
    if (memcmp(&X[I], &Y[I], sizeof(M128A)) != 0)
      return 0;
  return 1;
}

/* The values the runners put in the nonvolatile registers: rbx, rsi, rdi,
   r12, r13, r14 and r15 hold a repeated byte 0x11, 0x22, 0x33, 0x44, 0x55, 0x66
   and 0x77, and xmm6-xmm15 hold 0x86-0x8f in every byte. */
static unsigned long long bytes(unsigned Byte) {
  return 0x0101010101010101ull * Byte;
}

static int sentinelsMatch(const CONTEXT *C, unsigned long long Flags) {
  if (C->Rbx != bytes(0x11) || C->Rsi != bytes(0x22) || C->Rdi != bytes(0x33) ||
      C->R12 != bytes(0x44) || C->R15 != bytes(0x77))
    return 0;
  if (!(Flags & 1) && (C->R13 != bytes(0x55) || C->R14 != bytes(0x66)))
    return 0;
  const M128A *X = &C->Xmm6;
  for (unsigned I = 0; I != 10; ++I)
    if (X[I].Low != bytes(0x86 + I) ||
        (unsigned long long)X[I].High != bytes(0x86 + I))
      return 0;
  return 1;
}

/* stepRun, which calls the runner while single-stepping; set in main. */
static int (*volatile StepRunPtr)(const TestCase *, int);

/* Unwinds the frame that Ctx is in, as the exception dispatcher does. A
   function with no unwind data is a leaf: the return address is at RSP. */
static void unwindOne(CONTEXT *Ctx) {
  DWORD64 Base;
  PRUNTIME_FUNCTION FE = RtlLookupFunctionEntry(Ctx->Rip, &Base, NULL);
  if (!FE) {
    Ctx->Rip = *(const DWORD64 *)Ctx->Rsp;
    Ctx->Rsp += 8;
    return;
  }
  PVOID HandlerData;
  DWORD64 Establisher;
  RtlVirtualUnwind(UNW_FLAG_NHANDLER, Base, Ctx->Rip, FE, Ctx, &HandlerData,
                   &Establisher, NULL);
}

static int functionBegins(DWORD64 Pc, DWORD64 Expected) {
  DWORD64 Base;
  PRUNTIME_FUNCTION FE = RtlLookupFunctionEntry(Pc, &Base, NULL);
  return FE && Base + FE->BeginAddress == Expected;
}

/* The state the unwinder gives at a boundary inside a checked function. */
static int checkUnwind(const CONTEXT *At, LONG Step) {
  CONTEXT Ctx = *At;
  unwindOne(&Ctx);
  if (!functionBegins(Ctx.Rip, (DWORD64)CurrentCase->Run)) {
    printf("FAIL %s[%d] step %ld: rip %p: unwound to %p, not the runner\n",
           CurrentCase->Name, CurrentSelector, Step, (void *)At->Rip,
           (void *)Ctx.Rip);
    return 0;
  }
  if (!sameRegisters(&Ctx, &unwind_expected_context, CurrentCase->Flags)) {
    printf("FAIL %s[%d] step %ld: rip %p: nonvolatile registers differ after "
           "unwinding\n",
           CurrentCase->Name, CurrentSelector, Step, (void *)At->Rip);
    return 0;
  }
  /* The runner's own frame must be unwindable whatever RSP the unwinder gave,
     which is why callers of these conventions have a frame pointer. Its caller
     is stepRun. */
  unwindOne(&Ctx);
  if (!functionBegins(Ctx.Rip, (DWORD64)StepRunPtr)) {
    printf("FAIL %s[%d] step %ld: rip %p: the runner's caller (stepRun) was "
           "not found\n",
           CurrentCase->Name, CurrentSelector, Step, (void *)At->Rip);
    return 0;
  }
  return 1;
}

static LONG filter(EXCEPTION_POINTERS *EP) {
  if (EP->ExceptionRecord->ExceptionCode != EXCEPTION_SINGLE_STEP)
    return EXCEPTION_CONTINUE_SEARCH;
  CONTEXT *C = EP->ContextRecord;
  if (isCheckedPc(C->Rip)) {
    LONG Step = InterlockedIncrement(&Steps);
    if (CheckState && Step == 1 && !sentinelsMatch(C, CurrentCase->Flags)) {
      printf("FAIL %s[%d]: the nonvolatile registers were not the sentinels "
             "when the function under test was entered\n",
             CurrentCase->Name, CurrentSelector);
      ++Failures;
    }
    if (CheckState && !checkUnwind(C, Step))
      ++Failures;
    if (TargetStep > 0 && Step == TargetStep) {
      InterlockedIncrement(&Reached);
      return EXCEPTION_EXECUTE_HANDLER;
    }
  }
  /* Keep stepping only while the runner is being stepped. Clearing TF with
     popfq still traps once more, and re-arming it then would leave TF set
     after stepRun returns, so the next trap would be in code that is not
     covered by the __try. */
  if (Stepping)
    C->EFlags |= TRAP_FLAG;
  else
    C->EFlags &= ~TRAP_FLAG;
  return EXCEPTION_CONTINUE_EXECUTION;
}

static __declspec(noinline) int stepRun(const TestCase *TC, int Selector) {
  int Result = -1;
  Stepping = 1;
  __writeeflags(__readeflags() | TRAP_FLAG);
  Result = TC->Run(Selector);
  Stepping = 0;
  __writeeflags(__readeflags() & ~TRAP_FLAG);
  return Result;
}

/* Runs the case once. Mode 0: normally. Mode 1: single-stepping. */
static __declspec(noinline) int runOne(const TestCase *TC, int Selector,
                                       int Mode, LONG Target) {
  int Result = -1;
  CurrentCase = TC;
  CurrentSelector = Selector;
  Steps = 0;
  Reached = 0;
  TargetStep = Target;
  CheckState = Mode == 1 && Target == 0;
  if (Mode == 0) {
    Result = TC->Run(Selector);
  } else {
    __try {
      Result = stepRun(TC, Selector);
    } __except (filter(GetExceptionInformation())) {
      Stepping = 0;
      __writeeflags(__readeflags() & ~TRAP_FLAG);
    }
  }
  return Result;
}

int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);
  StepRunPtr = stepRun;
  sortFunctions();

  /* Things that make the results meaningless or the run die. */
  if (IsDebuggerPresent())
    printf("WARNING: running under a debugger, which receives the single-step "
           "exceptions before the program does\n");
  if (*(const unsigned char *)StepRunPtr == 0xE9)
    printf("WARNING: function addresses are jump thunks (an incremental "
           "link), so comparisons with the unwinder's results will fail\n");
  int Cases = 0;

  for (const TestCase *TC = unwind_test_cases; TC->Name; ++TC) {
    for (int Sel = 0; Sel != (int)TC->NumSelectors; ++Sel) {
      int Expected = (TC->TailMask >> Sel) & 1;
      ++Cases;

      printf("%s[%d]: normal run\n", TC->Name, Sel);
      int Before = Failures;
      int Result = runOne(TC, Sel, 0, 0);
      if (Result != Expected) {
        printf("FAIL %s[%d]: the tail call target saw wrong arguments (%d, "
               "expected %d)\n",
               TC->Name, Sel, Result, Expected);
        ++Failures;
      }
      if (!sentinelsMatch(&unwind_expected_context, TC->Flags)) {
        printf("FAIL %s[%d]: the runner did not have the sentinel values in "
               "the nonvolatile registers when it captured its context\n",
               TC->Name, Sel);
        ++Failures;
      }
      if (!sameRegisters(&unwind_actual_context, &unwind_expected_context,
                         TC->Flags)) {
        printf("FAIL %s[%d]: nonvolatile registers changed across the call\n",
               TC->Name, Sel);
        ++Failures;
      }

      printf("%s[%d]: single-stepping, unwinding at each step\n", TC->Name,
             Sel);
      Result = runOne(TC, Sel, 1, 0);
      LONG Total = Steps;
      if (Result != Expected) {
        printf("FAIL %s[%d]: wrong result when single-stepping (%d)\n",
               TC->Name, Sel, Result);
        ++Failures;
      }
      if (Total == 0) {
        printf("FAIL %s[%d]: no instruction of a checked function was "
               "stepped\n",
               TC->Name, Sel);
        ++Failures;
      }

      for (LONG K = 1; K <= Total; ++K) {
        printf("%s[%d]: exception dispatch at step %ld of %ld\n", TC->Name, Sel,
               K, Total);
        runOne(TC, Sel, 1, K);
        if (Reached != 1) {
          printf("FAIL %s[%d]: the handler was not reached at step %ld\n",
                 TC->Name, Sel, K);
          ++Failures;
        }
      }
      if (Failures == Before)
        printf("PASS %s[%d] (%ld steps)\n", TC->Name, Sel, Total);
    }
  }

  printf("%d run(s), %d failure(s)\n", Cases, Failures);
  return Failures != 0;
}
