//===-- ExternalFunctions.cpp - Implement External Functions --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file contains both code to deal with invoking "external" functions, but
//  also contains code that implements "exported" external functions.
//
//  External functions in the interpreter are implemented by
//  using the system's dynamic loader to look up the address of the function
//  we want to invoke.  If a function is found, then one of the
//  many lle_* wrapper functions in this file will translate its arguments from
//  GenericValues to the types the function is actually expecting, before the
//  function is called.
//
//===----------------------------------------------------------------------===//

#include "Interpreter.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Support/Streams.h"
#include "llvm/System/DynamicLibrary.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/ManagedStatic.h"
#include <csignal>
#include <cstdio>
#include <map>
#include <cmath>
#include <cstring>

#ifdef __linux__
#include <cxxabi.h>
#endif

using std::vector;

using namespace llvm;

typedef GenericValue (*ExFunc)(FunctionType *, const vector<GenericValue> &);
static ManagedStatic<std::map<const Function *, ExFunc> > Functions;
static std::map<std::string, ExFunc> FuncNames;

static Interpreter *TheInterpreter;

static char getTypeID(const Type *Ty) {
  switch (Ty->getTypeID()) {
  case Type::VoidTyID:    return 'V';
  case Type::IntegerTyID:
    switch (cast<IntegerType>(Ty)->getBitWidth()) {
      case 1:  return 'o';
      case 8:  return 'B';
      case 16: return 'S';
      case 32: return 'I';
      case 64: return 'L';
      default: return 'N';
    }
  case Type::FloatTyID:   return 'F';
  case Type::DoubleTyID:  return 'D';
  case Type::PointerTyID: return 'P';
  case Type::FunctionTyID:return 'M';
  case Type::StructTyID:  return 'T';
  case Type::ArrayTyID:   return 'A';
  case Type::OpaqueTyID:  return 'O';
  default: return 'U';
  }
}

// Try to find address of external function given a Function object.
// Please note, that interpreter doesn't know how to assemble a
// real call in general case (this is JIT job), that's why it assumes,
// that all external functions has the same (and pretty "general") signature.
// The typical example of such functions are "lle_X_" ones.
static ExFunc lookupFunction(const Function *F) {
  // Function not found, look it up... start by figuring out what the
  // composite function name should be.
  std::string ExtName = "lle_";
  const FunctionType *FT = F->getFunctionType();
  for (unsigned i = 0, e = FT->getNumContainedTypes(); i != e; ++i)
    ExtName += getTypeID(FT->getContainedType(i));
  ExtName += "_" + F->getName();

  ExFunc FnPtr = FuncNames[ExtName];
  if (FnPtr == 0)
    FnPtr = FuncNames["lle_X_"+F->getName()];
  if (FnPtr == 0)  // Try calling a generic function... if it exists...
    FnPtr = (ExFunc)(intptr_t)sys::DynamicLibrary::SearchForAddressOfSymbol(
            ("lle_X_"+F->getName()).c_str());
  if (FnPtr == 0)
    FnPtr = (ExFunc)(intptr_t)
      sys::DynamicLibrary::SearchForAddressOfSymbol(F->getName());
  if (FnPtr != 0)
    Functions->insert(std::make_pair(F, FnPtr));  // Cache for later
  return FnPtr;
}

GenericValue Interpreter::callExternalFunction(Function *F,
                                     const std::vector<GenericValue> &ArgVals) {
  TheInterpreter = this;

  // Do a lookup to see if the function is in our cache... this should just be a
  // deferred annotation!
  std::map<const Function *, ExFunc>::iterator FI = Functions->find(F);
  ExFunc Fn = (FI == Functions->end()) ? lookupFunction(F) : FI->second;
  if (Fn == 0) {
    cerr << "Tried to execute an unknown external function: "
         << F->getType()->getDescription() << " " << F->getName() << "\n";
    if (F->getName() == "__main")
      return GenericValue();
    abort();
  }

  // TODO: FIXME when types are not const!
  GenericValue Result = Fn(const_cast<FunctionType*>(F->getFunctionType()),
                           ArgVals);
  return Result;
}


//===----------------------------------------------------------------------===//
//  Functions "exported" to the running application...
//
extern "C" {  // Don't add C++ manglings to llvm mangling :)

// void putchar(ubyte)
GenericValue lle_X_putchar(FunctionType *FT, const vector<GenericValue> &Args){
  cout << ((char)Args[0].IntVal.getZExtValue()) << std::flush;
  return Args[0];
}

// void _IO_putc(int c, FILE* fp)
GenericValue lle_X__IO_putc(FunctionType *FT, const vector<GenericValue> &Args){
#ifdef __linux__
  _IO_putc((char)Args[0].IntVal.getZExtValue(), (FILE*) Args[1].PointerVal);
#else
  assert(0 && "Can't call _IO_putc on this platform");
#endif
  return Args[0];
}

// void atexit(Function*)
GenericValue lle_X_atexit(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  TheInterpreter->addAtExitHandler((Function*)GVTOP(Args[0]));
  GenericValue GV;
  GV.IntVal = 0;
  return GV;
}

// void exit(int)
GenericValue lle_X_exit(FunctionType *FT, const vector<GenericValue> &Args) {
  TheInterpreter->exitCalled(Args[0]);
  return GenericValue();
}

// void abort(void)
GenericValue lle_X_abort(FunctionType *FT, const vector<GenericValue> &Args) {
  raise (SIGABRT);
  return GenericValue();
}

// void *malloc(uint)
GenericValue lle_X_malloc(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 1 && "Malloc expects one argument!");
  assert(isa<PointerType>(FT->getReturnType()) && "malloc must return pointer");
  return PTOGV(malloc(Args[0].IntVal.getZExtValue()));
}

// void *calloc(uint, uint)
GenericValue lle_X_calloc(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 2 && "calloc expects two arguments!");
  assert(isa<PointerType>(FT->getReturnType()) && "calloc must return pointer");
  return PTOGV(calloc(Args[0].IntVal.getZExtValue(), 
                      Args[1].IntVal.getZExtValue()));
}

// void *calloc(uint, uint)
GenericValue lle_X_realloc(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 2 && "calloc expects two arguments!");
  assert(isa<PointerType>(FT->getReturnType()) &&"realloc must return pointer");
  return PTOGV(realloc(GVTOP(Args[0]), Args[1].IntVal.getZExtValue()));
}

// void free(void *)
GenericValue lle_X_free(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  free(GVTOP(Args[0]));
  return GenericValue();
}

// int atoi(char *)
GenericValue lle_X_atoi(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  GenericValue GV;
  GV.IntVal = APInt(32, atoi((char*)GVTOP(Args[0])));
  return GV;
}

// double pow(double, double)
GenericValue lle_X_pow(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 2);
  GenericValue GV;
  GV.DoubleVal = pow(Args[0].DoubleVal, Args[1].DoubleVal);
  return GV;
}

// double sin(double)
GenericValue lle_X_sin(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  GenericValue GV;
  GV.DoubleVal = sin(Args[0].DoubleVal);
  return GV;
}

// double cos(double)
GenericValue lle_X_cos(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  GenericValue GV;
  GV.DoubleVal = cos(Args[0].DoubleVal);
  return GV;
}

// double exp(double)
GenericValue lle_X_exp(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  GenericValue GV;
  GV.DoubleVal = exp(Args[0].DoubleVal);
  return GV;
}

// double sqrt(double)
GenericValue lle_X_sqrt(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  GenericValue GV;
  GV.DoubleVal = sqrt(Args[0].DoubleVal);
  return GV;
}

// double log(double)
GenericValue lle_X_log(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  GenericValue GV;
  GV.DoubleVal = log(Args[0].DoubleVal);
  return GV;
}

// double floor(double)
GenericValue lle_X_floor(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  GenericValue GV;
  GV.DoubleVal = floor(Args[0].DoubleVal);
  return GV;
}

#ifdef HAVE_RAND48

// double drand48()
GenericValue lle_X_drand48(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.empty());
  GenericValue GV;
  GV.DoubleVal = drand48();
  return GV;
}

// long lrand48()
GenericValue lle_X_lrand48(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.empty());
  GenericValue GV;
  GV.IntVal = APInt(32, lrand48());
  return GV;
}

// void srand48(long)
GenericValue lle_X_srand48(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  srand48(Args[0].IntVal.getZExtValue());
  return GenericValue();
}

#endif

// int rand()
GenericValue lle_X_rand(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.empty());
  GenericValue GV;
  GV.IntVal = APInt(32, rand());
  return GV;
}

// void srand(uint)
GenericValue lle_X_srand(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  srand(Args[0].IntVal.getZExtValue());
  return GenericValue();
}

// int puts(const char*)
GenericValue lle_X_puts(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  GenericValue GV;
  GV.IntVal = APInt(32, puts((char*)GVTOP(Args[0])));
  return GV;
}

// int sprintf(sbyte *, sbyte *, ...) - a very rough implementation to make
// output useful.
GenericValue lle_X_sprintf(FunctionType *FT, const vector<GenericValue> &Args) {
  char *OutputBuffer = (char *)GVTOP(Args[0]);
  const char *FmtStr = (const char *)GVTOP(Args[1]);
  unsigned ArgNo = 2;

  // printf should return # chars printed.  This is completely incorrect, but
  // close enough for now.
  GenericValue GV; 
  GV.IntVal = APInt(32, strlen(FmtStr));
  while (1) {
    switch (*FmtStr) {
    case 0: return GV;             // Null terminator...
    default:                       // Normal nonspecial character
      sprintf(OutputBuffer++, "%c", *FmtStr++);
      break;
    case '\\': {                   // Handle escape codes
      sprintf(OutputBuffer, "%c%c", *FmtStr, *(FmtStr+1));
      FmtStr += 2; OutputBuffer += 2;
      break;
    }
    case '%': {                    // Handle format specifiers
      char FmtBuf[100] = "", Buffer[1000] = "";
      char *FB = FmtBuf;
      *FB++ = *FmtStr++;
      char Last = *FB++ = *FmtStr++;
      unsigned HowLong = 0;
      while (Last != 'c' && Last != 'd' && Last != 'i' && Last != 'u' &&
             Last != 'o' && Last != 'x' && Last != 'X' && Last != 'e' &&
             Last != 'E' && Last != 'g' && Last != 'G' && Last != 'f' &&
             Last != 'p' && Last != 's' && Last != '%') {
        if (Last == 'l' || Last == 'L') HowLong++;  // Keep track of l's
        Last = *FB++ = *FmtStr++;
      }
      *FB = 0;

      switch (Last) {
      case '%':
        strcpy(Buffer, "%"); break;
      case 'c':
        sprintf(Buffer, FmtBuf, uint32_t(Args[ArgNo++].IntVal.getZExtValue()));
        break;
      case 'd': case 'i':
      case 'u': case 'o':
      case 'x': case 'X':
        if (HowLong >= 1) {
          if (HowLong == 1 &&
              TheInterpreter->getTargetData()->getPointerSizeInBits() == 64 &&
              sizeof(long) < sizeof(int64_t)) {
            // Make sure we use %lld with a 64 bit argument because we might be
            // compiling LLI on a 32 bit compiler.
            unsigned Size = strlen(FmtBuf);
            FmtBuf[Size] = FmtBuf[Size-1];
            FmtBuf[Size+1] = 0;
            FmtBuf[Size-1] = 'l';
          }
          sprintf(Buffer, FmtBuf, Args[ArgNo++].IntVal.getZExtValue());
        } else
          sprintf(Buffer, FmtBuf,uint32_t(Args[ArgNo++].IntVal.getZExtValue()));
        break;
      case 'e': case 'E': case 'g': case 'G': case 'f':
        sprintf(Buffer, FmtBuf, Args[ArgNo++].DoubleVal); break;
      case 'p':
        sprintf(Buffer, FmtBuf, (void*)GVTOP(Args[ArgNo++])); break;
      case 's':
        sprintf(Buffer, FmtBuf, (char*)GVTOP(Args[ArgNo++])); break;
      default:  cerr << "<unknown printf code '" << *FmtStr << "'!>";
        ArgNo++; break;
      }
      strcpy(OutputBuffer, Buffer);
      OutputBuffer += strlen(Buffer);
      }
      break;
    }
  }
  return GV;
}

// int printf(sbyte *, ...) - a very rough implementation to make output useful.
GenericValue lle_X_printf(FunctionType *FT, const vector<GenericValue> &Args) {
  char Buffer[10000];
  vector<GenericValue> NewArgs;
  NewArgs.push_back(PTOGV((void*)&Buffer[0]));
  NewArgs.insert(NewArgs.end(), Args.begin(), Args.end());
  GenericValue GV = lle_X_sprintf(FT, NewArgs);
  cout << Buffer;
  return GV;
}

static void ByteswapSCANFResults(const char *Fmt, void *Arg0, void *Arg1,
                                 void *Arg2, void *Arg3, void *Arg4, void *Arg5,
                                 void *Arg6, void *Arg7, void *Arg8) {
  void *Args[] = { Arg0, Arg1, Arg2, Arg3, Arg4, Arg5, Arg6, Arg7, Arg8, 0 };

  // Loop over the format string, munging read values as appropriate (performs
  // byteswaps as necessary).
  unsigned ArgNo = 0;
  while (*Fmt) {
    if (*Fmt++ == '%') {
      // Read any flag characters that may be present...
      bool Suppress = false;
      bool Half = false;
      bool Long = false;
      bool LongLong = false;  // long long or long double

      while (1) {
        switch (*Fmt++) {
        case '*': Suppress = true; break;
        case 'a': /*Allocate = true;*/ break;  // We don't need to track this
        case 'h': Half = true; break;
        case 'l': Long = true; break;
        case 'q':
        case 'L': LongLong = true; break;
        default:
          if (Fmt[-1] > '9' || Fmt[-1] < '0')   // Ignore field width specs
            goto Out;
        }
      }
    Out:

      // Read the conversion character
      if (!Suppress && Fmt[-1] != '%') { // Nothing to do?
        unsigned Size = 0;
        const Type *Ty = 0;

        switch (Fmt[-1]) {
        case 'i': case 'o': case 'u': case 'x': case 'X': case 'n': case 'p':
        case 'd':
          if (Long || LongLong) {
            Size = 8; Ty = Type::Int64Ty;
          } else if (Half) {
            Size = 4; Ty = Type::Int16Ty;
          } else {
            Size = 4; Ty = Type::Int32Ty;
          }
          break;

        case 'e': case 'g': case 'E':
        case 'f':
          if (Long || LongLong) {
            Size = 8; Ty = Type::DoubleTy;
          } else {
            Size = 4; Ty = Type::FloatTy;
          }
          break;

        case 's': case 'c': case '[':  // No byteswap needed
          Size = 1;
          Ty = Type::Int8Ty;
          break;

        default: break;
        }

        if (Size) {
          GenericValue GV;
          void *Arg = Args[ArgNo++];
          memcpy(&GV, Arg, Size);
          TheInterpreter->StoreValueToMemory(GV, (GenericValue*)Arg, Ty);
        }
      }
    }
  }
}

// int sscanf(const char *format, ...);
GenericValue lle_X_sscanf(FunctionType *FT, const vector<GenericValue> &args) {
  assert(args.size() < 10 && "Only handle up to 10 args to sscanf right now!");

  char *Args[10];
  for (unsigned i = 0; i < args.size(); ++i)
    Args[i] = (char*)GVTOP(args[i]);

  GenericValue GV;
  GV.IntVal = APInt(32, sscanf(Args[0], Args[1], Args[2], Args[3], Args[4],
                        Args[5], Args[6], Args[7], Args[8], Args[9]));
  ByteswapSCANFResults(Args[1], Args[2], Args[3], Args[4],
                       Args[5], Args[6], Args[7], Args[8], Args[9], 0);
  return GV;
}

// int scanf(const char *format, ...);
GenericValue lle_X_scanf(FunctionType *FT, const vector<GenericValue> &args) {
  assert(args.size() < 10 && "Only handle up to 10 args to scanf right now!");

  char *Args[10];
  for (unsigned i = 0; i < args.size(); ++i)
    Args[i] = (char*)GVTOP(args[i]);

  GenericValue GV;
  GV.IntVal = APInt(32, scanf( Args[0], Args[1], Args[2], Args[3], Args[4],
                        Args[5], Args[6], Args[7], Args[8], Args[9]));
  ByteswapSCANFResults(Args[0], Args[1], Args[2], Args[3], Args[4],
                       Args[5], Args[6], Args[7], Args[8], Args[9]);
  return GV;
}


// int clock(void) - Profiling implementation
GenericValue lle_i_clock(FunctionType *FT, const vector<GenericValue> &Args) {
  extern unsigned int clock(void);
  GenericValue GV; 
  GV.IntVal = APInt(32, clock());
  return GV;
}


//===----------------------------------------------------------------------===//
// String Functions...
//===----------------------------------------------------------------------===//

// int strcmp(const char *S1, const char *S2);
GenericValue lle_X_strcmp(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 2);
  GenericValue Ret;
  Ret.IntVal = APInt(32, strcmp((char*)GVTOP(Args[0]), (char*)GVTOP(Args[1])));
  return Ret;
}

// char *strcat(char *Dest, const char *src);
GenericValue lle_X_strcat(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 2);
  assert(isa<PointerType>(FT->getReturnType()) &&"strcat must return pointer");
  return PTOGV(strcat((char*)GVTOP(Args[0]), (char*)GVTOP(Args[1])));
}

// char *strcpy(char *Dest, const char *src);
GenericValue lle_X_strcpy(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 2);
  assert(isa<PointerType>(FT->getReturnType()) &&"strcpy must return pointer");
  return PTOGV(strcpy((char*)GVTOP(Args[0]), (char*)GVTOP(Args[1])));
}

static GenericValue size_t_to_GV (size_t n) {
  GenericValue Ret;
  if (sizeof (size_t) == sizeof (uint64_t)) {
    Ret.IntVal = APInt(64, n);
  } else {
    assert (sizeof (size_t) == sizeof (unsigned int));
    Ret.IntVal = APInt(32, n);
  }
  return Ret;
}

static size_t GV_to_size_t (GenericValue GV) {
  size_t count;
  if (sizeof (size_t) == sizeof (uint64_t)) {
    count = (size_t)GV.IntVal.getZExtValue();
  } else {
    assert (sizeof (size_t) == sizeof (unsigned int));
    count = (size_t)GV.IntVal.getZExtValue();
  }
  return count;
}

// size_t strlen(const char *src);
GenericValue lle_X_strlen(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  size_t strlenResult = strlen ((char *) GVTOP (Args[0]));
  return size_t_to_GV (strlenResult);
}

// char *strdup(const char *src);
GenericValue lle_X_strdup(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  assert(isa<PointerType>(FT->getReturnType()) && "strdup must return pointer");
  return PTOGV(strdup((char*)GVTOP(Args[0])));
}

// char *__strdup(const char *src);
GenericValue lle_X___strdup(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  assert(isa<PointerType>(FT->getReturnType()) &&"_strdup must return pointer");
  return PTOGV(strdup((char*)GVTOP(Args[0])));
}

// void *memset(void *S, int C, size_t N)
GenericValue lle_X_memset(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 3);
  size_t count = GV_to_size_t (Args[2]);
  assert(isa<PointerType>(FT->getReturnType()) && "memset must return pointer");
  return PTOGV(memset(GVTOP(Args[0]), uint32_t(Args[1].IntVal.getZExtValue()), 
                      count));
}

// void *memcpy(void *Dest, void *src, size_t Size);
GenericValue lle_X_memcpy(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 3);
  assert(isa<PointerType>(FT->getReturnType()) && "memcpy must return pointer");
  size_t count = GV_to_size_t (Args[2]);
  return PTOGV(memcpy((char*)GVTOP(Args[0]), (char*)GVTOP(Args[1]), count));
}

// void *memcpy(void *Dest, void *src, size_t Size);
GenericValue lle_X_memmove(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 3);
  assert(isa<PointerType>(FT->getReturnType()) && "memmove must return pointer");
  size_t count = GV_to_size_t (Args[2]);
  return PTOGV(memmove((char*)GVTOP(Args[0]), (char*)GVTOP(Args[1]), count));
}

//===----------------------------------------------------------------------===//
// IO Functions...
//===----------------------------------------------------------------------===//

// getFILE - Turn a pointer in the host address space into a legit pointer in
// the interpreter address space.  This is an identity transformation.
#define getFILE(ptr) ((FILE*)ptr)

// FILE *fopen(const char *filename, const char *mode);
GenericValue lle_X_fopen(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 2);
  assert(isa<PointerType>(FT->getReturnType()) && "fopen must return pointer");
  return PTOGV(fopen((const char *)GVTOP(Args[0]),
                     (const char *)GVTOP(Args[1])));
}

// int fclose(FILE *F);
GenericValue lle_X_fclose(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  GenericValue GV;
  GV.IntVal = APInt(32, fclose(getFILE(GVTOP(Args[0]))));
  return GV;
}

// int feof(FILE *stream);
GenericValue lle_X_feof(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  GenericValue GV;

  GV.IntVal = APInt(32, feof(getFILE(GVTOP(Args[0]))));
  return GV;
}

// size_t fread(void *ptr, size_t size, size_t nitems, FILE *stream);
GenericValue lle_X_fread(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 4);
  size_t result;

  result = fread((void*)GVTOP(Args[0]), GV_to_size_t (Args[1]),
                 GV_to_size_t (Args[2]), getFILE(GVTOP(Args[3])));
  return size_t_to_GV (result);
}

// size_t fwrite(const void *ptr, size_t size, size_t nitems, FILE *stream);
GenericValue lle_X_fwrite(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 4);
  size_t result;

  result = fwrite((void*)GVTOP(Args[0]), GV_to_size_t (Args[1]),
                  GV_to_size_t (Args[2]), getFILE(GVTOP(Args[3])));
  return size_t_to_GV (result);
}

// char *fgets(char *s, int n, FILE *stream);
GenericValue lle_X_fgets(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 3);
  return PTOGV(fgets((char*)GVTOP(Args[0]), Args[1].IntVal.getZExtValue(),
                     getFILE(GVTOP(Args[2]))));
}

// FILE *freopen(const char *path, const char *mode, FILE *stream);
GenericValue lle_X_freopen(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 3);
  assert(isa<PointerType>(FT->getReturnType()) &&"freopen must return pointer");
  return PTOGV(freopen((char*)GVTOP(Args[0]), (char*)GVTOP(Args[1]),
                       getFILE(GVTOP(Args[2]))));
}

// int fflush(FILE *stream);
GenericValue lle_X_fflush(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  GenericValue GV;
  GV.IntVal = APInt(32, fflush(getFILE(GVTOP(Args[0]))));
  return GV;
}

// int getc(FILE *stream);
GenericValue lle_X_getc(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  GenericValue GV;
  GV.IntVal = APInt(32, getc(getFILE(GVTOP(Args[0]))));
  return GV;
}

// int _IO_getc(FILE *stream);
GenericValue lle_X__IO_getc(FunctionType *F, const vector<GenericValue> &Args) {
  return lle_X_getc(F, Args);
}

// int fputc(int C, FILE *stream);
GenericValue lle_X_fputc(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 2);
  GenericValue GV;
  GV.IntVal = APInt(32, fputc(Args[0].IntVal.getZExtValue(), 
                              getFILE(GVTOP(Args[1]))));
  return GV;
}

// int ungetc(int C, FILE *stream);
GenericValue lle_X_ungetc(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 2);
  GenericValue GV;
  GV.IntVal = APInt(32, ungetc(Args[0].IntVal.getZExtValue(), 
                               getFILE(GVTOP(Args[1]))));
  return GV;
}

// int ferror (FILE *stream);
GenericValue lle_X_ferror(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  GenericValue GV;
  GV.IntVal = APInt(32, ferror (getFILE(GVTOP(Args[0]))));
  return GV;
}

// int fprintf(FILE *,sbyte *, ...) - a very rough implementation to make output
// useful.
GenericValue lle_X_fprintf(FunctionType *FT, const vector<GenericValue> &Args) {
  assert(Args.size() >= 2);
  char Buffer[10000];
  vector<GenericValue> NewArgs;
  NewArgs.push_back(PTOGV(Buffer));
  NewArgs.insert(NewArgs.end(), Args.begin()+1, Args.end());
  GenericValue GV = lle_X_sprintf(FT, NewArgs);

  fputs(Buffer, getFILE(GVTOP(Args[0])));
  return GV;
}

// int __cxa_guard_acquire (__guard *g);
GenericValue lle_X___cxa_guard_acquire(FunctionType *FT, 
                                       const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
  GenericValue GV;
#ifdef __linux__
  GV.IntVal = APInt(32, __cxxabiv1::__cxa_guard_acquire (
                          (__cxxabiv1::__guard*)GVTOP(Args[0])));
#else
  assert(0 && "Can't call __cxa_guard_acquire on this platform");
#endif
  return GV;
}

// void __cxa_guard_release (__guard *g);
GenericValue lle_X___cxa_guard_release(FunctionType *FT, 
                                       const vector<GenericValue> &Args) {
  assert(Args.size() == 1);
#ifdef __linux__
  __cxxabiv1::__cxa_guard_release ((__cxxabiv1::__guard*)GVTOP(Args[0]));
#else
  assert(0 && "Can't call __cxa_guard_release on this platform");
#endif
  return GenericValue();
}

} // End extern "C"


void Interpreter::initializeExternalFunctions() {
  FuncNames["lle_X_putchar"]      = lle_X_putchar;
  FuncNames["lle_X__IO_putc"]     = lle_X__IO_putc;
  FuncNames["lle_X_exit"]         = lle_X_exit;
  FuncNames["lle_X_abort"]        = lle_X_abort;
  FuncNames["lle_X_malloc"]       = lle_X_malloc;
  FuncNames["lle_X_calloc"]       = lle_X_calloc;
  FuncNames["lle_X_realloc"]      = lle_X_realloc;
  FuncNames["lle_X_free"]         = lle_X_free;
  FuncNames["lle_X_atoi"]         = lle_X_atoi;
  FuncNames["lle_X_pow"]          = lle_X_pow;
  FuncNames["lle_X_sin"]          = lle_X_sin;
  FuncNames["lle_X_cos"]          = lle_X_cos;
  FuncNames["lle_X_exp"]          = lle_X_exp;
  FuncNames["lle_X_log"]          = lle_X_log;
  FuncNames["lle_X_floor"]        = lle_X_floor;
  FuncNames["lle_X_srand"]        = lle_X_srand;
  FuncNames["lle_X_rand"]         = lle_X_rand;
#ifdef HAVE_RAND48
  FuncNames["lle_X_drand48"]      = lle_X_drand48;
  FuncNames["lle_X_srand48"]      = lle_X_srand48;
  FuncNames["lle_X_lrand48"]      = lle_X_lrand48;
#endif
  FuncNames["lle_X_sqrt"]         = lle_X_sqrt;
  FuncNames["lle_X_puts"]         = lle_X_puts;
  FuncNames["lle_X_printf"]       = lle_X_printf;
  FuncNames["lle_X_sprintf"]      = lle_X_sprintf;
  FuncNames["lle_X_sscanf"]       = lle_X_sscanf;
  FuncNames["lle_X_scanf"]        = lle_X_scanf;
  FuncNames["lle_i_clock"]        = lle_i_clock;

  FuncNames["lle_X_strcmp"]       = lle_X_strcmp;
  FuncNames["lle_X_strcat"]       = lle_X_strcat;
  FuncNames["lle_X_strcpy"]       = lle_X_strcpy;
  FuncNames["lle_X_strlen"]       = lle_X_strlen;
  FuncNames["lle_X___strdup"]     = lle_X___strdup;
  FuncNames["lle_X_memset"]       = lle_X_memset;
  FuncNames["lle_X_memcpy"]       = lle_X_memcpy;
  FuncNames["lle_X_memmove"]      = lle_X_memmove;

  FuncNames["lle_X_fopen"]        = lle_X_fopen;
  FuncNames["lle_X_fclose"]       = lle_X_fclose;
  FuncNames["lle_X_feof"]         = lle_X_feof;
  FuncNames["lle_X_fread"]        = lle_X_fread;
  FuncNames["lle_X_fwrite"]       = lle_X_fwrite;
  FuncNames["lle_X_fgets"]        = lle_X_fgets;
  FuncNames["lle_X_fflush"]       = lle_X_fflush;
  FuncNames["lle_X_fgetc"]        = lle_X_getc;
  FuncNames["lle_X_getc"]         = lle_X_getc;
  FuncNames["lle_X__IO_getc"]     = lle_X__IO_getc;
  FuncNames["lle_X_fputc"]        = lle_X_fputc;
  FuncNames["lle_X_ungetc"]       = lle_X_ungetc;
  FuncNames["lle_X_fprintf"]      = lle_X_fprintf;
  FuncNames["lle_X_freopen"]      = lle_X_freopen;

  FuncNames["lle_X___cxa_guard_acquire"] = lle_X___cxa_guard_acquire;
  FuncNames["lle_X____cxa_guard_release"] = lle_X___cxa_guard_release;
}

