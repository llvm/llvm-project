# The WinCall Calling Convention Draft

## What WinCall is

WinCall is an x86-64 **calling convention for Windows and other PE/COFF
targets** (Windows, Cygwin, MSYS and UEFI). It is spelled ``x86_wincallcc``
in LLVM IR and corresponds to ``CallingConv::X86_WinCall`` (CC ID 128).

Its purpose is to exploit the 16 additional general-purpose registers
(R16-R31) introduced by Intel APX. On a conventional Windows x64 ABI only
four integer arguments can be passed in registers; WinCall doubles that to
eight by using R16-R19 as argument registers. It also relaxes the aggregate
passing and return rules of the Microsoft x64 ABI so that C++ types such as
``std::span`` (a ``{pointer, size}`` pair) and other small aggregates can
travel in registers.

WinCall does **not** replace any existing ABI. Existing Windows APIs keep
their ``stdcall``, ``cdecl`` and ``fastcall`` conventions; the convention is
opt-in and only affects code that is explicitly built for it.

## When WinCall is used

WinCall applies in two situations:

1. It is the **default calling convention** of the ``x86_64apx`` **PE/COFF**
   targets: ``x86_64apx-windows-msvc``, ``x86_64apx-windows-gnu``, and the
   other PE targets that share the Windows ABI conventions — Cygwin
   (``x86_64apx-pc-windows-cygnus``), MSYS (``x86_64apx-pc-windows-msys``)
   and UEFI (``x86_64apx-unknown-uefi``).
   (``X86_64TargetInfo::getDefaultCallingConv`` returns ``CC_WinCall`` for
   ``getTriple().isWindowsAPX()``, which is true for any ``x86_64apx`` triple
   whose OS is Windows or UEFI.)

2. It can be selected per-function with the ``wincall`` attribute
   (``__attribute__((wincall))``, ``__wincall``, ``_wincall``,
   ``[[gnu::wincall]]``), which maps to ``CC_WinCall`` in the frontend.
   WinCall is the APX-aware Microsoft ABI, so ``wincall`` already implies the
   MS x64 ABI — there is no need to also write ``ms_abi``. (``ms_abi`` alone is
   still the classic MS x64 convention on non-APX targets, while on
   ``x86_64apx`` targets ``ms_abi`` alone implies ``wincall``.)

WinCall is **x86-64 only**. On 32-bit x86 or on any non-x86 target the
attribute is ignored with a warning
(``X86TargetInfo::checkCallingConvention`` does not accept it there).

## The x86_64apx sub-architecture

The ``x86_64apx`` triple component is a *sub-architecture* of ``x86_64``,
modelled on how ``arm64ec`` is a sub-architecture of ``aarch64``:

- ``Triple::X86_64SubArch_apx`` records it; ``getArch()`` still returns
  ``Triple::x86_64``.
- ``Triple::isX86_64APX()`` is true for any ``x86_64apx-*`` triple.
- ``Triple::isWindowsAPX()`` is true for ``x86_64apx-*`` triples whose OS is
  Windows or UEFI (i.e. the PE/COFF targets); only these default to WinCall.
- The name round-trips through ``Triple::getArchName()`` /
  ``Triple::parseSubArch()``.

The sub-architecture also turns on the APX instruction-set extensions. Both
the clang frontend (``X86TargetInfo::initFeatureMap``) and the LLVM backend
(``X86_MC::ParseX86Triple``) enable:

```
+egpr,+push2pop2,+ppx,+ndd,+ccmp,+nf,+zu,+jmpabs
```

for ``x86_64apx`` targets. There is no single ``apxf`` feature bit; APX is
modelled as this set of independent features. The backend must enable them
from the triple itself (not just from the frontend), because backend passes
size their register tables from ``X86RegisterInfo::getNumSupportedRegs()``,
which returns the full register count only when ``egpr`` is enabled. If the
registers were not enabled, WinCall would still assign R16-R19 as argument
registers, and passes such as ``LiveVariables`` would index out of bounds.

## Argument passing

### Integer and pointer arguments

The first **eight** integer-class arguments are passed in registers:

| arg # | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    |
|-------|------|------|------|------|------|------|------|------|
| i8    | CL   | DL   | R8B  | R9B  | R16B | R17B | R18B | R19B |
| i16   | CX   | DX   | R8W  | R9W  | R16W | R17W | R18W | R19W |
| i32   | ECX  | EDX  | R8D  | R9D  | R16D | R17D | R18D | R19D |
| i64   | RCX  | RDX  | R8   | R9   | R16  | R17  | R18  | R19  |

``i1`` arguments are promoted to ``i8`` first. Remaining integer arguments go
on the stack in 8-byte, 8-byte-aligned slots.

This list is used by ``bool``, all integer types, pointers, ``__m64``,
``__int128``/``__uint128_t`` (which is split into two i64s), and
``std::float128_t``.

### Floating-point, SIMD and vector arguments

The first **eight** FP/SIMD arguments are passed in XMM registers (or
YMM/ZMM for wider vectors), **independently** of the integer registers:

| arg # | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    |
|-------|------|------|------|------|------|------|------|------|
| f16/f32/f64, 128-bit vectors | XMM0 | XMM1 | XMM2 | XMM3 | XMM4 | XMM5 | XMM6 | XMM7 |
| 256-bit vectors (``__m256``) | YMM0 | YMM1 | YMM2 | YMM3 | YMM4 | YMM5 | YMM6 | YMM7 |
| 512-bit vectors (``__m512``) | ZMM0 | ZMM1 | ZMM2 | ZMM3 | ZMM4 | ZMM5 | ZMM6 | ZMM7 |

The 256-bit rules require AVX, the 512-bit rules require AVX-512. Stack slots
are 16 bytes for 128-bit vectors, 32 bytes for 256-bit vectors and 64 bytes
for 512-bit vectors, aligned to their size.

### Independent integer/FP allocation

Unlike the Microsoft x64 ABI, which pairs each integer register with an XMM
register and *skips* the partner when the other class is used, WinCall
allocates the integer and FP register lists **independently**. A function
``f(int, double, int, double, int, double, ...)`` therefore uses
ECX, EDX, R8D, ... for its ints and XMM0, XMM1, XMM2, ... for its doubles
with no skipping.

This is safe across wincall/non-wincall calls because each function's
convention is fixed at compile time, and the ``@win`` symbol decoration (see
below) lets the linker catch mismatches.

### Aggregates

Clang's ``WinX86_64ABIInfo::classify`` implements the aggregate rules for
``CC_WinCall`` (this is a frontend rule layered on top of the IR-level
convention):

- A record of up to **32 bytes** is passed **directly in registers** (not by
  pointer/sret like the MS x64 ABI): a record of up to 64 bits is coerced to
  an integer of its size and uses **one** GPR; a larger record (e.g. 16, 24
  or 32 bytes) is **expanded** into its 8-byte parts. A 4-``size_t`` struct
  therefore travels in RCX, RDX, R8, R9, and a 3-``size_t`` (24-byte) struct
  in RCX, RDX, R8.
- **Empty records** (``struct empty {}``) consume **no register slots**;
  ``classify`` returns ``ABIArgInfo::getIgnore()`` for them.
- Records larger than 32 bytes, records with a flexible array member, and
  non-trivial C++ records (per ``getRecordArgABI``) are passed by reference.
  Note that this means C++ classes with user-declared or user-provided
  destructors or copy/move constructors (e.g. ``std::string``,
  ``std::vector``, ``std::unique_ptr``) are **not** passed in registers —
  they are passed by pointer regardless of size.
- ``f80`` (long double) is passed by pointer.
- ``__int128`` is split into two GPRs.
- Complex types and member pointers are handled as in the MS x64 ABI.

## Return values

``RetCC_X86_Win64_WinCall``:

| Return type | Registers |
|-------------|-----------|
| Scalar (i8/i16/i32/i64), ``__m64`` | RAX (first value), RDX, RCX, R8 |
| Scalar up to 128 bits (``__int128``, ``__uint128_t``, ``std::float128_t``) | RAX (low) + RDX (high) |
| f16/f32/f64 and 128-bit vectors | XMM0 (first value), XMM1 |
| 256-bit vectors | YMM0 |
| 512-bit vectors | ZMM0 |
| Aggregates up to 32 bytes | RAX, RDX, RCX, R8 (expanded) |
| Aggregates larger than 32 bytes | hidden pointer (sret) |

Empty records consume no return register. The state of unused bits in RAX or
XMM0 is undefined.

## Callee-saved registers

WinCall keeps the Windows x64 caller-saved model. The callee-saved set is
``CSR_Win64_APX`` (used whenever ``egpr`` is enabled, since the frontend
always enables it for ``x86_64apx``):

```
RBX, RBP, RDI, RSI, R12, R13, R14, R15, R30, R31, XMM6-XMM15
```

R30 and R31 are APX registers preserved in addition to the standard Windows
x64 set.

## Symbol decoration: the ``@win`` suffix

Every WinCall function gets a ``@win`` suffix appended to its symbol name, so
that the linker and loader can detect a caller and callee compiled with
mismatched calling conventions. This mirrors the ``@N`` parameter-size
suffix of ``stdcall`` on i386, but uses a fixed tag that cannot collide with
``stdcall``'s ``@0``/``@16`` decorations.

The decoration is applied uniformly across all three name manglings
(implemented in ``clang/lib/AST/Mangle.cpp`` for C, and via the suffix check
in ``clang/lib/AST/MicrosoftMangle.cpp``; Itanium C++ names are decorated by
the same ``@win`` suffix path):

| Language / ABI | Symbol |
|----------------|--------|
| C              | ``foo@win`` |
| C++ MS ABI     | ``?foo@@YAXH@Z@win`` |
| C++ Itanium ABI | ``_Z3fooi@win`` |

Additionally, the Itanium mangling encodes WinCall in function-pointer types
as the vendor extended qualifier ``U7wincall``:

```cpp
using W = void (__attribute__((wincall)) *)(int);
template <typename T> T id(T x);
W w;
// _Z2idIPU7wincallFviEET_S2_
```

## Section alignment

For ``x86_64apx`` targets the clang driver passes a **64 KiB section
alignment** to the linker by default, so that the OS can use 64 KiB pages
(matching the page size of the NVIDIA Grace CPU) for code and data built with
this convention:

| Toolchain | Default flag |
|-----------|--------------|
| MSVC ``link.exe`` | ``/section-alignment:0x10000`` and ``/driver`` |
| MinGW ``ld``      | ``--section-alignment=0x10000`` |

The flags are added by the driver in ``clang/lib/Driver/ToolChains/MSVC.cpp``
and ``MinGW.cpp``, not by the assembler. If the user passes their own
section-alignment ``-Wl`` flag, the driver's default is suppressed in favour
of the user's value.

The 64 KiB section alignment is only meaningful for binaries loaded by an OS
using 64 KiB pages, so it is **not** added automatically for the Cygwin and
UEFI toolchains (which use the generic GCC driver and ``lld-link``
respectively); those are typically firmware or runtime-loader images that want
a conventional section alignment. The stack alignment guarantee below still
applies to them, since it is a property of the generated code, not of the
linked image.

## Stack alignment

On ``isWindowsAPX()`` targets (``x86_64apx-windows``, ``x86_64apx-cygnus``,
``x86_64apx-msys`` and ``x86_64apx-uefi``) the stack is kept **64-byte
aligned** at every call site (``X86Subtarget`` sets the stack alignment to 64
for ``isWindowsAPX()`` targets, instead of the 16-byte alignment of the
classic Windows ABI). This is a deliberate part of the WinCall ABI: it means
the backend can use aligned 64-byte moves (``vmovaps``/``vmovdqa64``) for
AVX-512 ZMM spills and aligned stack slots without dynamic stack realignment.

This matters in practice because the classic Windows x64 ABI only guarantees
16-byte stack alignment, which is not enough for 64-byte ZMM registers — this
is why GCC still cannot support AVX-512 on Windows correctly. WinCall's 64-byte
guarantee removes that limitation. Only ``isWindowsAPX()`` targets get the
64-byte alignment; other ``x86_64apx`` targets (e.g. ``x86_64apx-linux``) keep
the 16-byte default, and a user-supplied ``-mstack-alignment`` still overrides
it.

## Building a DLL that works with both WinCall and the classic ABI

A function's ABI is decided at the *call boundary*, not inside the function.
A single DLL binary can therefore serve both WinCall callers and classic
(``stdcall`` / MS x64 / plain MinGW) callers, but only by exposing **two entry
points per function** — one per ABI — rather than one function with a
compromise ABI. The three things that differ between the two worlds each need
their own solution:

| Aspect        | WinCall                        | Classic ABI                |
|---------------|--------------------------------|----------------------------|
| Symbol        | ``foo@win``                    | ``foo``                    |
| Calling conv. | 8 GPRs, XMM0-7, aggregates in registers | 4 GPRs, XMM0-3, sret/byval |
| ``long double`` | f64, 8 bytes                 | f80, 16 bytes              |

### Symbol names

Export **both** names for the same body: ``foo`` and ``foo@win``. On COFF this
is two export entries pointing at the same RVA (or a one-line asm alias). The
``@win`` name serves WinCall callers; the plain name serves classic callers.

### Calling convention: forwarding thunks

WinCall and the classic convention genuinely disagree on registers and stack.
The standard technique is a thin **forwarding thunk** per ABI that converts
the argument placement and jumps to a single implementation compiled with
WinCall:

```asm
foo@win:            ; WinCall ABI: args in RCX,RDX,R8,R9,R16-R19, XMM0-7
        jmp foo_impl

foo:                ; classic ABI: args in RCX,RDX,R8,R9, XMM0-3
        ...         ; convert the small subset that differs
        jmp foo_impl
```

This is the same machinery as C++ ABI thunks or
``-fdefault-calling-conv`` plus a per-function calling-convention attribute.
In practice the thunk is written in assembly, or the exported function is
marked with the classic attribute (e.g. ``__attribute__((stdcall))``) and the
compiler emits the conversion.

### ``long double``

This is the one genuine limitation: a single binary stores one
representation. The options are:

1. **Keep ``long double`` out of the exported interface** (recommended). Most
   Windows DLL APIs use ``double``/``int``/pointers/structs, so WinCall's f64
   ``long double`` is invisible to classic callers and the dual-ABI DLL works
   with both worlds.
2. **Keep the classic 16-byte f80 ``long double`` in exported functions** that
   must cross the boundary, while WinCall-internal code uses f64. This needs a
   per-function (or per-TU) ``long double`` layout switch, so the exported
   surface matches classic callers.
3. **Pick f64 everywhere** and accept that only WinCall callers may use
   ``long double`` in the API. Simple, but classic callers passing f80 will
   misbehave on those functions.

The ``long double`` size change therefore does **not** by itself break
dual-ABI DLLs — the calling-convention and symbol differences already require
the dual-entry-thunk design, and ``long double`` only matters if it is part of
the exported interface.

## Relation to "herbceptions" (deterministic exceptions)

WinCall is designed so that Herb Sutter's proposed zero-overhead
deterministic exceptions (P0709, "herbceptions") can represent ``std::error``
as a two-register value passed/returned in RAX (domain pointer) and RDX
(code), with the carry flag (CF) as the success/failure discriminant. This is
the design intent of the convention; the discriminant-lowering support in
LLVM is independent of WinCall and is not part of this calling convention
itself.

## Examples

The following examples show how common C++ types are passed and returned
under WinCall (``x86_64apx-windows-msvc``, optimized output).

### Empty objects

An empty object (``struct empty {}``) consumes **no register slots**. The
``int`` that follows it takes the first GPR (ECX):

```c
struct empty {};

__attribute__((wincall)) void f(struct empty e, int x) { sink(x); }
```

```asm
f@win:
        callq   sink@win        # x is forwarded from ECX; e took no register
```

An empty object as a return type also uses no register. This makes C++ types
that contain empty base classes or members cheaper to pass and return than
under the MS x64 ABI.

### ``__uint128_t`` (and ``__int128_t``, ``std::float128_t``)

A 128-bit integer is split into **two GPRs**: passed in RCX (low) + RDX
(high), returned in RAX (low) + RDX (high).

```c
__attribute__((wincall)) __uint128_t f(__uint128_t v);
```

```asm
f@win:
        movq    %rcx, %rax      # low 64 bits: RCX -> RAX
        retq                    # high 64 bits already in RDX
```

The same two-register rule applies to ``__int128_t`` and to
``std::float128_t``.

### ``std::span`` (two-word aggregates)

A ``{pointer, size}`` aggregate such as ``std::span`` or
``std::string_view`` is 16 bytes, so it is passed in **two** GPRs: the
pointer in RCX and the length in RDX, and returned in RAX (pointer) + RDX
(length).

```c
struct span { void *base; unsigned long long len; };

__attribute__((wincall)) void f(struct span s);
__attribute__((wincall)) struct span g(void);
```

```asm
        callq   g@win           # returns {RAX = base, RDX = len}
        movq    %rax, %rcx      # pass base in RCX (len already in RDX)
        callq   f@win
```

Under the MS x64 ABI this same ``std::span`` argument would be passed by
pointer; WinCall makes it a zero-cost, purely-register argument.

### Four-word aggregates

A plain 4-``size_t`` struct (32 bytes) is passed in **four** GPRs (RCX,
RDX, R8, R9) and returned in RAX, RDX, RCX, R8. See the FAQ below for the
exact code. (C++ classes such as ``std::string`` or ``std::vector`` that
have a non-trivial destructor or copy/move constructor are *not* passed in
registers — they are passed by pointer; see the Aggregates section above.)

## FAQ

### R30 and R31 are callee-saved — can they still be used as scratch registers?

Yes. R30 and R31 are callee-saved registers under the Windows APX ABI, so a
callee may freely use them as scratch registers as long as it preserves them
across the call. The compiler does exactly that: when a WinCall function uses
R30 or R31 it pushes them in the prologue and pops them in the epilogue using
the APX ``pushp``/``popp`` instructions:

```asm
use_r31@win:
        pushp   %r31           # preserve R31
        movq    %rcx, %r31     # use R31 as a scratch register
        callq   g
        popp    %r31           # restore R31
        retq
```

This mirrors how the other callee-saved registers (RBX, RDI, RSI, R12-R15,
XMM6-XMM15) work on the standard Windows x64 ABI. The only restriction is in
functions that call ``setjmp``/``longjmp``: because the Windows unwinder
cannot restore the APX extended registers across a jump, clang reserves
R30/R31 there (and warns on large functions), so they are not allocated.

### Does a 4-pointer struct split into four registers?

Yes. A struct of four pointers (32 bytes) is classified by
``WinX86_64ABIInfo::classify`` as a direct record: it is returned in
``RAX, RDX, RCX, R8`` and passed as an argument in ``RCX, RDX, R8, R9`` —
four separate GPR slots, one per 8-byte field. At ``-O2`` the words move
directly from the return registers to the argument registers with no stack
round-trip:

```asm
caller@win:
        callq   make@win       # returns the 4-pointer struct in RAX/RDX/RCX/R8
        movq    %rcx, %r9      # save words 3,4 in scratch regs
        movq    %r8,  %r10
        movq    %rax, %rcx     # arg 1
        movq    %rdx, %rdx     # arg 2
        movq    %r9,  %r8      # arg 3
        movq    %r10, %r9      # arg 4
        callq   take@win
```

(With optimization disabled the four words are spilled to and reloaded from
the stack frame between the two calls, but the argument registers are still
RCX, RDX, R8, R9.)

An integer argument following the struct is placed in the next free GPR
(R16). This is how a four-word aggregate such as a 4-``size_t`` struct
travels entirely in registers under WinCall, unlike the MS x64 ABI which
would pass it by pointer. (C++ classes with non-trivial destructors or
copy/move constructors are exempt from this register passing; see the
Aggregates section above.)

## Implementation notes

- IR calling convention: ``x86_wincallcc`` / ``CallingConv::X86_WinCall``
  (CC ID 128).
- TableGen conventions in ``llvm/lib/Target/X86/X86CallingConv.td``:
  ``CC_X86_Win64_WinCall`` (arguments) and ``RetCC_X86_Win64_WinCall``
  (returns), dispatched from the root ``CC_X86_64`` convention.
- Frontend ABI rules: ``WinX86_64ABIInfo::classify`` in
  ``clang/lib/CodeGen/Targets/X86.cpp``.
- Default CC: ``X86_64TargetInfo::getDefaultCallingConv`` in
  ``clang/lib/Basic/Targets/X86.h``.
- Symbol decoration: ``clang/lib/AST/Mangle.cpp`` and
  ``clang/lib/AST/MicrosoftMangle.cpp``; vendor qualifier in
  ``clang/lib/AST/ItaniumMangle.cpp``.
- Triples: ``Triple::X86_64SubArch_apx`` in ``llvm/lib/TargetParser/Triple.*``.
- Driver flags: ``clang/lib/Driver/ToolChains/MSVC.cpp`` and ``MinGW.cpp``.
- Register tables must include the APX registers when ``egpr`` is enabled;
  see ``X86RegisterInfo::getNumSupportedRegs``.
