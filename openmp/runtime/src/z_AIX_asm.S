//  z_AIX_asm.S:  - microtasking routines specifically
//                  written for Power platforms running AIX OS

//
////===----------------------------------------------------------------------===//
////
//// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
//// See https://llvm.org/LICENSE.txt for license information.
//// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
////
////===----------------------------------------------------------------------===//
//

// -----------------------------------------------------------------------
// macros
// -----------------------------------------------------------------------

#include "kmp_config.h"

#if KMP_OS_AIX
//------------------------------------------------------------------------
// int
// __kmp_invoke_microtask( void (*pkfn) (int *gtid, int *tid, ...),
//                         int gtid, int tid,
//                         int argc, void *p_argv[]
// #if OMPT_SUPPORT
//                         ,
//                         void **exit_frame_ptr
// #endif
//                       ) {
// #if OMPT_SUPPORT
//   *exit_frame_ptr = OMPT_GET_FRAME_ADDRESS(0);
// #endif
//
//   (*pkfn)( & gtid, & tid, p_argv[0], ... );
//
// // FIXME: This is done at call-site and can be removed here.
// #if OMPT_SUPPORT
//   *exit_frame_ptr = 0;
// #endif
//
//   return 1;
// }
//
// parameters:
//   r3: pkfn
//   r4: gtid
//   r5: tid
//   r6: argc
//   r7: p_argv
//   r8: &exit_frame
//
// return:  r3 (always 1/TRUE)
//

#if KMP_ARCH_PPC64_XCOFF

    .globl  __kmp_invoke_microtask[DS]
    .globl  .__kmp_invoke_microtask
    .align  4
    .csect __kmp_invoke_microtask[DS],3
    .vbyte  8, .__kmp_invoke_microtask
    .vbyte  8, TOC[TC0]
    .vbyte  8, 0
    .csect .text[PR],2
    .machine "pwr7"
.__kmp_invoke_microtask:


// -- Begin __kmp_invoke_microtask
// mark_begin;

// We need to allocate a stack frame large enough to hold all of the parameters
// on the stack for the microtask plus what this function needs. That's 48
// bytes under the XCOFF64 ABI, plus max(64, 8*(2 + argc)) for
// the parameters to the microtask (gtid, tid, argc elements of p_argv),
// plus 8 bytes to store the values of r4 and r5, and 8 bytes to store r31.
// With OMP-T support, we need an additional 8 bytes to save r30 to hold
// a copy of r8.
// Stack offsets relative to stack pointer:
//   r31: -8, r30: -16, gtid: -20, tid: -24

    mflr 0
    std 31, -8(1)      # Save r31 to the stack
    std 0, 16(1)       # Save LR to the linkage area

// This is unusual because normally we'd set r31 equal to r1 after the stack
// frame is established. In this case, however, we need to dynamically compute
// the stack frame size, and so we keep a direct copy of r1 to access our
// register save areas and restore the r1 value before returning.
    mr 31, 1

// Compute the size of the "argc" portion of the parameter save area.
// The parameter save area is always at least 64 bytes long (i.e. 8 regs)
// The microtask has (2 + argc) parameters, so if argc <= 6, we need to
// to allocate 8*6 bytes, not 8*argc.
    li 0, 6
    cmpwi 0, 6, 6
    iselgt 0, 6, 0     # r0 = (argc > 6)? argc : 6
    sldi 0, 0, 3       # r0 = 8 * max(argc, 6)

// Compute the size necessary for the local stack frame.
// 88 = 48 + 4 (for r4) + 4 (for r5) + 8 (for r31) + 8 (for OMP-T r30) +
//      8 (parameter gtid) + 8 (parameter tid)
    li 12, 88
    add 12, 0, 12
    neg 12, 12

// We need to make sure that the stack frame stays aligned (to 16 bytes).
    li 0, -16
    and 12, 0, 12

// Establish the local stack frame.
    stdux 1, 1, 12

#if OMPT_SUPPORT
    std 30, -16(31)    # Save r30 to the stack
    std 1, 0(8)
    mr 30, 8
#endif

// Store gtid and tid to the stack because they're passed by reference to the microtask.
    stw 4, -20(31)     # Save gtid to the stack
    stw 5, -24(31)     # Save tid to the stack

    mr 12, 6           # r12 = argc
    mr 4, 7            # r4 = p_argv

    cmpwi 0, 12, 1
    blt 0, .Lcall      # if (argc < 1) goto .Lcall

    ld 5, 0(4)         # r5 = p_argv[0]

    cmpwi 0, 12, 2
    blt 0, .Lcall      # if (argc < 2) goto .Lcall

    ld 6, 8(4)         # r6 = p_argv[1]

    cmpwi 0, 12, 3
    blt 0, .Lcall      # if (argc < 3) goto .Lcall

    ld 7, 16(4)        # r7 = p_argv[2]

    cmpwi 0, 12, 4
    blt 0, .Lcall      # if (argc < 4) goto .Lcall

    ld 8, 24(4)        # r8 = p_argv[3]

    cmpwi 0, 12, 5
    blt 0, .Lcall      # if (argc < 5) goto .Lcall

    ld 9, 32(4)        # r9 = p_argv[4]

    cmpwi 0, 12, 6
    blt 0, .Lcall      # if (argc < 6) goto .Lcall

    ld 10, 40(4)       # r10 = p_argv[5]

    cmpwi 0, 12, 7
    blt 0, .Lcall      # if (argc < 7) goto .Lcall

// There are more than 6 microtask parameters, so we need to store the
// remainder to the stack.
    addi 12, 12, -6    # argc -= 6
    mtctr 12

// These are set to 8 bytes before the first desired store address (we're using
// pre-increment loads and stores in the loop below). The parameter save area
// for the microtask begins 48 + 8*8 == 112 bytes above r1 for XCOFF64.
    addi 4, 4, 40      # p_argv = p_argv + 5
                       # (i.e. skip the 5 elements we already processed)
    addi 12, 1, 104    # r12 = stack offset (112 - 8)

.Lnext:
    ldu 0, 8(4)
    stdu 0, 8(12)
    bdnz .Lnext

.Lcall:
    std 2, 40(1)     # Save the TOC pointer to the linkage area
// Load the actual function address from the function descriptor.
    ld 12, 0(3)      # Function address
    ld 2, 8(3)       # TOC pointer
    ld 11, 16(3)     # Environment pointer

    addi 3, 31, -20  # r3 = &gtid
    addi 4, 31, -24  # r4 = &tid

    mtctr 12         # CTR = function address
    bctrl            # Branch to CTR
    ld 2, 40(1)      # Restore TOC pointer from linkage area

#if OMPT_SUPPORT
    li 3, 0
    std 3, 0(30)
#endif

    li 3, 1

#if OMPT_SUPPORT
    ld 30, -16(31)   # Restore r30 from the saved value on the stack
#endif

    mr 1, 31
    ld 31, -8(1)     # Restore r31 from the saved value on the stack
    ld 0, 16(1)
    mtlr 0           # Restore LR from the linkage area
    blr              # Branch to LR

#else  // KMP_ARCH_PPC_XCOFF

    .globl  __kmp_invoke_microtask[DS]
    .globl  .__kmp_invoke_microtask
    .align  4
    .csect __kmp_invoke_microtask[DS],2
    .vbyte  4, .__kmp_invoke_microtask
    .vbyte  4, TOC[TC0]
    .vbyte  4, 0
    .csect .text[PR],2
    .machine "pwr7"
.__kmp_invoke_microtask:


// -- Begin __kmp_invoke_microtask
// mark_begin;

// We need to allocate a stack frame large enough to hold all of the parameters
// on the stack for the microtask plus what this function needs. That's 24
// bytes under the XCOFF ABI, plus max(32, 8*(2 + argc)) for
// the parameters to the microtask (gtid, tid, argc elements of p_argv),
// plus 8 bytes to store the values of r4 and r5, and 4 bytes to store r31.
// With OMP-T support, we need an additional 4 bytes to save r30 to hold
// a copy of r8.
// Stack offsets relative to stack pointer:
//   r31: -4, r30: -8, gtid: -12, tid: -16

    mflr 0
    stw 31, -4(1)      # Save r31 to the stack
    stw 0, 8(1)        # Save LR to the linkage area

// This is unusual because normally we'd set r31 equal to r1 after the stack
// frame is established. In this case, however, we need to dynamically compute
// the stack frame size, and so we keep a direct copy of r1 to access our
// register save areas and restore the r1 value before returning.
    mr 31, 1

// Compute the size of the "argc" portion of the parameter save area.
// The parameter save area is always at least 32 bytes long (i.e. 8 regs)
// The microtask has (2 + argc) parameters, so if argc <= 6, we need to
// to allocate 4*6 bytes, not 4*argc.
    li 0, 6
    cmpwi 0, 6, 6
    iselgt 0, 6, 0     # r0 = (argc > 6)? argc : 6
    slwi 0, 0, 2       # r0 = 4 * max(argc, 6)

// Compute the size necessary for the local stack frame.
// 56 = 32 + 4 (for r4) + 4 (for r5) + 4 (for r31) + 4 (for OMP-T r30) +
//      4 (parameter gtid) + 4 (parameter tid)
    li 12, 56
    add 12, 0, 12
    neg 12, 12

// We need to make sure that the stack frame stays aligned (to 16 bytes).
    li 0, -16
    and 12, 0, 12

// Establish the local stack frame.
    stwux 1, 1, 12

#if OMPT_SUPPORT
    stw 30, -8(31)     # Save r30 to the stack
    stw 1, 0(8)
    mr 30, 8
#endif

// Store gtid and tid to the stack because they're passed by reference to the microtask.
    stw 4, -12(31)     # Save gtid to the stack
    stw 5, -16(31)     # Save tid to the stack

    mr 12, 6           # r12 = argc
    mr 4, 7            # r4 = p_argv

    cmpwi 0, 12, 1
    blt 0, .Lcall      # if (argc < 1) goto .Lcall

    lwz 5, 0(4)        # r5 = p_argv[0]

    cmpwi 0, 12, 2
    blt 0, .Lcall      # if (argc < 2) goto .Lcall

    lwz 6, 4(4)        # r6 = p_argv[1]

    cmpwi 0, 12, 3
    blt 0, .Lcall      # if (argc < 3) goto .Lcall

    lwz 7, 8(4)        # r7 = p_argv[2]

    cmpwi 0, 12, 4
    blt 0, .Lcall      # if (argc < 4) goto .Lcall

    lwz 8, 12(4)       # r8 = p_argv[3]

    cmpwi 0, 12, 5
    blt 0, .Lcall      # if (argc < 5) goto .Lcall

    lwz 9, 16(4)       # r9 = p_argv[4]

    cmpwi 0, 12, 6
    blt 0, .Lcall      # if (argc < 6) goto .Lcall

    lwz 10, 20(4)      # r10 = p_argv[5]

    cmpwi 0, 12, 7
    blt 0, .Lcall      # if (argc < 7) goto .Lcall

// There are more than 6 microtask parameters, so we need to store the
// remainder to the stack.
    addi 12, 12, -6    # argc -= 6
    mtctr 12

// These are set to 4 bytes before the first desired store address (we're using
// pre-increment loads and stores in the loop below). The parameter save area
// for the microtask begins 24 + 4*8 == 56 bytes above r1 for XCOFF.
    addi 4, 4, 20      # p_argv = p_argv + 5
                       # (i.e. skip the 5 elements we already processed)
    addi 12, 1, 52     # r12 = stack offset (56 - 4)

.Lnext:
    lwzu 0, 4(4)
    stwu 0, 4(12)
    bdnz .Lnext

.Lcall:
    stw 2, 20(1)     # Save the TOC pointer to the linkage area
// Load the actual function address from the function descriptor.
    lwz 12, 0(3)     # Function address
    lwz 2, 4(3)      # TOC pointer
    lwz 11, 8(3)     # Environment pointer

    addi 3, 31, -12  # r3 = &gtid
    addi 4, 31, -16  # r4 = &tid

    mtctr 12         # CTR = function address
    bctrl            # Branch to CTR
    lwz 2, 20(1)     # Restore TOC pointer from linkage area

#if OMPT_SUPPORT
    li 3, 0
    stw 3, 0(30)
#endif

    li 3, 1

#if OMPT_SUPPORT
    lwz 30, -8(31)   # Restore r30 from the saved value on the stack
#endif

    mr 1, 31
    lwz 31, -4(1)    # Restore r31 from the saved value on the stack
    lwz 0, 8(1)
    mtlr 0           # Restore LR from the linkage area
    blr              # Branch to LR

#endif // KMP_ARCH_PPC64_XCOFF

.Lfunc_end0:
    .vbyte  4, 0x00000000           # Traceback table begin
    .byte   0x00                    # Version = 0
    .byte   0x09                    # Language = CPlusPlus
    .byte   0x20                    # -IsGlobalLinkage, -IsOutOfLineEpilogOrPrologue
                                    # +HasTraceBackTableOffset, -IsInternalProcedure
                                    # -HasControlledStorage, -IsTOCless
                                    # -IsFloatingPointPresent
                                    # -IsFloatingPointOperationLogOrAbortEnabled
    .byte   0x61                    # -IsInterruptHandler, +IsFunctionNamePresent, +IsAllocaUsed
                                    # OnConditionDirective = 0, -IsCRSaved, +IsLRSaved
    .byte   0x80                    # +IsBackChainStored, -IsFixup, NumOfFPRsSaved = 0
#if OMPT_SUPPORT
    .byte   0x02                    # -HasExtensionTable, -HasVectorInfo, NumOfGPRsSaved = 2
    .byte   0x06                    # NumberOfFixedParms = 6
#else
    .byte   0x01                    # -HasExtensionTable, -HasVectorInfo, NumOfGPRsSaved = 1
    .byte   0x05                    # NumberOfFixedParms = 5
#endif
    .byte   0x01                    # NumberOfFPParms = 0, +HasParmsOnStack
    .vbyte  4, 0x00000000           # Parameter type = i, i, i, i, i
    .vbyte  4, .Lfunc_end0-.__kmp_invoke_microtask # Function size
    .vbyte  2, 0x0016               # Function name len = 22
    .byte   "__kmp_invoke_microtask" # Function Name
    .byte   0x1f                    # AllocaRegister = 31
                                    # -- End function

// -- End  __kmp_invoke_microtask

// Support for unnamed common blocks.

    .comm .gomp_critical_user_, 32, 3
#if KMP_ARCH_PPC64_XCOFF
    .csect __kmp_unnamed_critical_addr[RW],3
#else
    .csect __kmp_unnamed_critical_addr[RW],2
#endif
    .globl __kmp_unnamed_critical_addr[RW]
    .ptr .gomp_critical_user_

// -- End unnamed common block

    .toc

#endif // KMP_OS_AIX
