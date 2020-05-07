# Propeller 2 LLVM Backend

The goal of this project is to write an LLVM backend to generate code for the Propeller 2 microcontroller. Need to do the following things from a high level (in no particular order)

Aside: This entire wiki is mostly stream of consciousness writing, so there might be contradictions. It's definitely not a clean document.

1. ~~figure out how to add a new target machine in the LLVM environment~~
    - https://llvm.org/docs/WritingAnLLVMBackend.html#preliminaries seems to have some starting points
1. ~~define a register format to use the COG memory register space~~ (see below)
1. ~~edit this machine to translate a basic program that is just main returning a constant.~~
    - minimum requires defining out the calling convention, ~~load~~, ~~store~~, and ~~ret instruction lowering~~.
    - the same page as above should get us there.
    - some great resources (though on opposite ends of the spectrum of sparse and dense):
        - https://jonathan2251.github.io/lbd/llvmstructure.html
        - https://llvm.org/devmtg/2012-04-12/Slides/Workshops/Anton_Korobeynikov.pdf
        - https://llvm.org/devmtg/2014-04/PDFs/Talks/Building%20an%20LLVM%20backend.pdf
1. ~~edit this machine to generate PASM code for basic ALU operations~~
    - expand to do all math instructions and provide custom nodes for operations propeller doesn't natively support
1. ~~edit this machine to support basic use of special registers (OUTx, DIRx, and INx)~~
    - currently only possible via function calls, eventually want to be able to write them directly.
1. ~~edit this machine to add some basic startup code to start the cog and execute a blinking LED program~~
    - what as actually needed here was to implement the linker backend and write a simple linker script, p2.ld
1. ~~add more propeller instructions to support branching~~
1. ~~implement calling conventions to call functions (especially recursion)~~
    - ~implement passing arguments by registers~
    - ~implement passing arguments by stack~
    - ~implement passing byval arguments (structs and classes)~
    - ~implement variable argument functions~
        - ~I think this will require spooling up the C standard library~ it won't
1. ~add support for starting cogs at specific memory location~
    - ~implement basic cog starting for cogs that do not require a stack~
    - ~implement including setq to pass the stack pointer and then assign the stack pointer to that value. Some startup code will be needed.~
1. expand on the rest of the propeller instruction set
1. create clang extensions for special directives and being able to write directly to I/O regsiters, specify functions and variables as going into hub, cog, or LUT memory, etc.
1. allow the COG attribute used by the linker to be read by LLVM and specify the correct relocation based on that.
1. port the necessary functions from the c standard library to make c/c++ useful.
    - port over propgcc library for this.
1. Figure out what's needed for classes
    - ~test basic classes that start static functions in new cogs.~
    - see if dynamic allocation will actually work
1. Expand assembly parser to support wc/wz/wcz and conditional modifiers on instructions
1. Random TODOs
    - ~change spilling/restoring callee saved regs using pusha/popa instead of rdlong/wrlong. would convert 3 instructions/register save into 1.~ implemented using special immediates into wrlong for PTRA (since pusha/popa is just an alias for wrlong/rdlong with an immediate)
    - ~change frame pointer elimination to instead of subtracting an offset to use the special form of rdlong/wrlong on ptra (page 60 of datasheet)~ this implementation is what actually does the above functionality.
    - ~implement "libcalls" for signed division, multiplication, etc and functions that can live in cog memory for speed.~ the implementation is a little fragile, since it relies on lib calls being a different type of symbol than normal function calls. Need to test if `extern` functions will have the same behavior and accidentally get relocated as lib functions. If that happens, will need to create a custom MCSymbol for cog based symbols.
    - clean up all the build warnings and get rid of a lot of extra commented code that's currently there.

The high level of how this will work:
1. use clang to compile c/c++ source into LLVM's IR language. Eventually any LLVM front end should work
1. use the custom backend (the goal of this project, using MIPS, AVR, and ARC targets as references) will convert the LLVM IR code to PASM-flavored LLVM instructions.
1. ~~use fastspin (or whatever Parallax's official assembler will be) to compile the assembly code into an executable elf to load.~~ Compile the PASM to an elf/binary directly from LLVM using clang and lld.

In principle, all the plumbing should already exist to run clang with the correct arguments and get an loadable binary out.

## Why are you writing yet another compiler for the Propeller 2? Fastspin works great

Yes, fastspin is great, but we need more.

Propeller (and Propeller 2) are power chips that can do A LOT of a small, simple, and power efficient package. The high flexibility allows it to be used in a very wide variety of applications without having to include a lot of support hardware. As such, it should be used widely in industry, but it's not (I work in aeropsace where a multicore chip like this would be extremely useful, yet no-one I work with has even heard of it). I think there are several reasons for its lack of adoption, but one of the biggest ones is the lack of a modern toolchain and lack of modern language support. Propeller 1 addressed this with PropGCC, but it was several years after the release of Propeller 1 and built around GCC 4, which is outdated in the modern world. Additionally, there appears to be a game of chicken going on between Parallax and the Propeller community, where Parallax is focused on Spin and the development of Propeller hardware (which is where their focus shoudl be right now), so they are hoping the community steps up (again) and develops the tools they desire, while the community is hoping to see something official come out and not put too much effort into developing something that might be pushed aside by an "official" toolchain. (As a general note, I'm not trying to start a debate about that here, it's just my observation). As a result, we have a few toolchain that are not quite good enough by industry standards (sorry to those who work on them, I know these tools take a lot of work and I do appreciate all the work that has been put in so far) that don't fully support C/C++ (like fastspin), and some that do have full C++ support, but do not support the full functionality of Propeller hardware (like RISCV-P2), and some in between, like p2gcc, which is more or less a bandaid for make use of PropGCC for Propeller 2 (p2gcc also doesn't support the most "standard" P2 library which makes code developed with it not very portable). While these are excellent tools to demonstrate the capabilities of the hardware, they make developing scalable products difficult if not impossible. There has also been several requests for various language support (microPython, Arduino, Rust, etc), all of which will require developing a compiler for Propeller's architecture.

This project aims to solve all of the problems listed above. LLVM is a modern toolchain used by many companies around the world, developed at Berkley a while back and supported primarily by Apple at this point. It has an intermediate representation that frontends (such as clang for C/C++/Objective C) compile down to, and target specific backends the compile the IR down to target machine instructions. The majority of the work to add a new backend is baked into LLVM as it, and the P2 target is another backend (same as the existing x86, AArch64, MIPS, AVR, MSP430, etc etc backends) that provides basic information (such as regsiters, instruction encoding, and ABI information) to connect the dots between the various compiler passes that LLVM does. Once complete, it will provide access to the full functionality of several langauges for Propeller.

I am developing this project with two main goals in mind:
1. create as much backward compatability as possible with PropGCC projects. This won't be completely possible due to a few differences, but the hope is that porting those P1 projects to P2 will be easy.
2. provide a tool that the community finds useful, regardless of adoption by Parallax as a formal tool. I know there's been some gripe on forums about the community's hard work not being adopted as much as people hope, but I am not pushing this for formal adoption. We'll just see what happens.

## Getting Started
See README.md, with the following notes:
- when running `cmake`, run `cmake -G "Unix Makefiles" -DLLVM_ENABLE_PROJECTS="lld;clang" -DCMAKE_INSTALL_PREFIX=<install dir> -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=P2 ../llvm` set the install directory to something useful, like `/opt/p2llvm/`
- building for the first time will take quite a bit of time, ~20 min. make sure to run `make -j<n>` to run on `n` cores
- to run one of the examples in p2_dev_tests, run in two steps (from p2_dev_tests/)
    - ~`../build/bin/clang -target mips-unknown-linux-gnu -S -c <file>.cpp -emit-llvm -o <file>.ll` (Eventually need to create a P2 target in clang). This compiles C down to LLVM IR language.~
    - ~`../build/bin/llc -march=p2 -debug -filetype=asm <file>.ll -o test2.s`. This will output an assembly file with P2 instructions. There's still work to be done to clean it up into something compilable with fastspin. Eventually should be able to compile PASM directly to objects to remove the need for fastspin entirely.~
    - except for the first few examples (which provide instructions in the header, and honestly have no purpose for being run), just change the target test in the `Makefile` and run `make`.
- The presentation by Anton Korobeynikov above gives a pretty high level but informative overview of how a backend works and the various components. This along with the other documents linked above help form a complete picture of how stuff works.
- This has all the definitions for instruction info creation: https://github.com/llvm-mirror/llvm/blob/master/include/llvm/Target/TargetSelectionDAG.td

## Cog Layout
The simplest way to make LLVM compatible with Propeller is to divide the cog memory into various sections for various compiler features. This section defines that layout.

### Register Definition
Cog memory is 512 longs. The last 16 are the special registers (0x1f0 - 0x1ff), so we will use the previous 16 registers (0x1e0 - 0x1ef) as general purpose registers for the compiler. So r0 = 0x1e0, r1 = 0x1e1, etc. PTRA is used as the stack pointer

### Stack
We'll need a stack for calling functions, etc. Ideally, the first 256 longs of the look-up RAM will be a fixed stack for calls and such. So, the stack starts at 0x0 and grows up to 0x0ff. This requires some extra work, so initially we'll just use hub RAM so that a typical architecture's load/store instructions can translate directly to rdlong/wrlong for memory read/writes, but eventually I'll add distinction between memories so that stack operations use rdlut/wrlut, and normal memory operations needing the hub RAM will use rdlong/wrlong.

The stack is organized starting at address 0x00300 and can continue to 0x003ff before begining to use program space. The stack will always grow up and each frame grows up within the stack. We won't have a distinct frame pointer, in each frame we should read in the current stack pointer and offset it locally to the desired stack slot. We use PTRA as the stack pointer, and PTRB as a scratch register for offseting the PTRA.

The remaining cog RAM should be used as a cache for loops, common functions, any math functions we want a fast implementation of, etc. Eventually, there should be a way to specify a function or variable to be cached in cog RAM so it never needs to be fetched from the hub. This will likely need a clang extension. This is also going to be a very nice to have as it might be difficult to specificy which cog a function should be in. So the thing we do now is specify functions to go into all cog memories (via the COG attribute).

## Calling Convention

### Passing Arguments
- All 8 and 16 bit arguments are promoted to 32 bits (if via registers)
- registers r0-r3 are used to pass first 4 arguments, remaining arguments are passed via the stack
- byval arguments will be passed by pointer via registers, after a local copy is made by the caller. A pointer to where to write a result will be passed as well. They can also be passed via stack, LLVM can make that decision without us worrying about it.

### Return Value
- Functions will return values using r15 and r14 (to easily support 64-bit returns)
- byvals will be returned via the pointer provided by the caller

When calling a function, the caller will increment `PTRA` to allocate space for the arguments that require stack space. These are stored in descending order, with formal arguemnts at the end/top of the frame, followed by any variable arguments. It will then call the function (using Propeller's CALLA instruction). This pushes the PC to PTRA. The callee will allocate the stack space it needs for it's local variables, saved registers and return values. This it will do its stuff and before returning (using the RETA instruction), it will de-allocate the stack space.

## Program and memory organization

This section describes how a program should be organized in hub memory. Below is a simple diagram.

| 0x00100                 | 0x00300     | 0x00400                        | 0x7fffff (or 0x7cffff?)          |
|-------------------------|-------------|--------------------------------|----------------------------------|
| Startup/COG memory code | Cog 0 stack | Start of generic program space | End of memory (on Rev B silicon) |

Hub memory 0x00300-0x003ff will be used as stack space for the cog 0 cog, as well as contain the startup code needed (like setting up the UART interface, etc). We put 0x00100 as the startup code because special registers live at 0x00000 (I can't find documentation on what these are, but 0x14 has clock frequency, 0x18 has clock mode, etc). We start the program space at 0x400. Starting a new cog requires setting PTRA (using SETQ) to the start of the stack space to use for that cog. The first stack slot should store the pointer to the function to run, the second stack slot should store an optional parameter that will be the first argument into the function. A new cog should always start by copying the startup code at 0x00100 and executing it.

The re-usable startup code should also contain all the basic libcall functions, like signed mul/div, and other basic and common functions that don't have direct instructions.

The end of the memory space (not yet described in detail) will be bss/rodata for static data and heap space for dynamic allocation, but dynamic allocation should be kept to a minimum (as with most embedded systems).

### Startup code

Right now, the only thing the startup code does is setup the stack pointer for the cog and jump to the starting function of the cog. If the cog is cog 0, this is hardcoded to set `ptra` to 0x00300 and jump to 0x00400. Otherwise, it pulls the initial stack pointer value and jump location from PTRA.

## Linking/Loading Programs

The linker script (p2.ld) does 2 things:
1. place `main()` at the start of .text, before anything else
1. place `_start()` at 0x00100 so that it is the first thing that executes
1. place .text at 0x00400, the first hub execution mode address.

The resulting elf can use Eric Smith's loadp2 (which supports elf binaries) and can be loaded onto the P2 Eval Board.

### Libraries

There are two libraries necessary to really round out the functionality of Propeller:
- The Propeller Standard Library
    - This is interfaced with via a standard propeller.h and include defintions to access hardware, start new cogs, startup code, etc.
- The C Standard Library
    - Exactly what it sounds like. An implementation of the C standard library for Propeller. Unlike the propgcc library, propeller specific things don't live in this library (like propeller.h).

The starting point of the Propeller Standard Library exists in p2_dev_tests/p2lib. This needs A LOT of work still and community agreement on what it should do.

There are a few things that intersect between the two Libraries (such as I/O FILE drivers for serial, etc). TBD on where this will go or how to implement it portably, but that problem is outside of the scope of this project (for now).

### Known Issues

- Jump tables don't work. Make sure to compile with -fno-jump-tables or else switch statements won't work.
- There's an issue somewhere with relocating symbols stored in .rodata (i.e if you try to do something like `struct_t a = {a_global_value}`), `a_global_value` won't get re-located correctly by the linker and the resulting code doesn't work. workaround is to explicitly set the value in a to `a_global_value`, i.e. `a.val = a_global_value`.
- ~mod operator gets lowered to multiplication/division/subtraction, which seems to sometimes not return the correct value. For instance, if writing a printf implementation, and we want to get the character for a given digit, we might do something like `"0123456789abcdef"[n % base]`. `n % base` seems to have some high bits set to 1 (seemingly random). I believe this is a result of how multiplication is done and the issue is the representation of the mul instruction in the backend. to be explored more.~ this is fixed by not using mul or muls for multiplication but qmul.
- likely related to above, don't have a way to distinguish between mul and qmul uses. should use mul if both operands are i16 zero-extended to i32, qmul otherwise. Probably some llvm pattern could be used, because there's no way to know (at compile time) if a register will have a zero-extended i16 other than very specific cases (such as using the result of a i16 zextload).
- something isn't correct with how functions are added to section (or how sections are referenced by the linker) cause trying to run with gc-sections removes all sections except main
