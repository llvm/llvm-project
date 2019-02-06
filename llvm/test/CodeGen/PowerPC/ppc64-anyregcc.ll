; RUN: llc -verify-machineinstrs < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; CHECK-LABEL: test:
; CHECK: {{^}}.L[[test_BEGIN:.*]]:{{$}}

; CHECK-LABEL: property_access1:
; CHECK: {{^}}.L[[property_access1_BEGIN:.*]]:{{$}}

; CHECK-LABEL: property_access2:
; CHECK: {{^}}.L[[property_access2_BEGIN:.*]]:{{$}}

; CHECK-LABEL: property_access3:
; CHECK: {{^}}.L[[property_access3_BEGIN:.*]]:{{$}}

; CHECK-LABEL: anyreg_test1:
; CHECK: {{^}}.L[[anyreg_test1_BEGIN:.*]]:{{$}}

; CHECK-LABEL: anyreg_test2:
; CHECK: {{^}}.L[[anyreg_test2_BEGIN:.*]]:{{$}}

; CHECK-LABEL: patchpoint_spilldef:
; CHECK: {{^}}.L[[patchpoint_spilldef_BEGIN:.*]]:{{$}}

; CHECK-LABEL: patchpoint_spillargs:
; CHECK: {{^}}.L[[patchpoint_spillargs_BEGIN:.*]]:{{$}}


; Stackmap Header: no constants - 6 callsites
; CHECK-LABEL: .section	.llvm_stackmaps
; CHECK-NEXT:  __LLVM_StackMaps:
; Header
; CHECK-NEXT:   .byte 3
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 0
; Num Functions
; CHECK-NEXT:   .long 8
; Num LargeConstants
; CHECK-NEXT:   .long 0
; Num Callsites
; CHECK-NEXT:   .long 8

; Functions and stack size
; CHECK-NEXT:   .quad test
; CHECK-NEXT:   .quad 128
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad property_access1
; CHECK-NEXT:   .quad 128
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad property_access2
; CHECK-NEXT:   .quad 128
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad property_access3
; CHECK-NEXT:   .quad 128
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad anyreg_test1
; CHECK-NEXT:   .quad 144
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad anyreg_test2
; CHECK-NEXT:   .quad 144
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad patchpoint_spilldef
; CHECK-NEXT:   .quad 256
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad patchpoint_spillargs
; CHECK-NEXT:   .quad 288
; CHECK-NEXT:   .quad 1


; test
; CHECK:  .long   .L{{.*}}-.L[[test_BEGIN]]
; CHECK-NEXT:   .short  0
; 3 locations
; CHECK-NEXT:   .short  3
; Loc 0: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 4
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 1: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 4
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 2: Constant 3
; CHECK-NEXT:   .byte 4
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 3
define i64 @test() nounwind ssp uwtable {
entry:
  call anyregcc void (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.void(i64 0, i32 40, i8* null, i32 2, i32 1, i32 2, i64 3)
  ret i64 0
}

; property access 1 - %obj is an anyreg call argument and should therefore be in a register
; CHECK:  .long   .L{{.*}}-.L[[property_access1_BEGIN]]
; CHECK-NEXT:   .short 0
; 2 locations
; CHECK-NEXT:   .short  2
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 1: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
define i64 @property_access1(i8* %obj) nounwind ssp uwtable {
entry:
  %f = inttoptr i64 281474417671919 to i8*
  %ret = call anyregcc i64 (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.i64(i64 1, i32 40, i8* %f, i32 1, i8* %obj)
  ret i64 %ret
}

; property access 2 - %obj is an anyreg call argument and should therefore be in a register
; CHECK:  .long   .L{{.*}}-.L[[property_access2_BEGIN]]
; CHECK-NEXT:   .short 0
; 2 locations
; CHECK-NEXT:   .short  2
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 1: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
define i64 @property_access2() nounwind ssp uwtable {
entry:
  %obj = alloca i64, align 8
  %f = inttoptr i64 281474417671919 to i8*
  %ret = call anyregcc i64 (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.i64(i64 2, i32 40, i8* %f, i32 1, i64* %obj)
  ret i64 %ret
}

; property access 3 - %obj is a frame index
; CHECK:  .long   .L{{.*}}-.L[[property_access3_BEGIN]]
; CHECK-NEXT:   .short 0
; 2 locations
; CHECK-NEXT:   .short  2
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 1: Direct FP - 8
; CHECK-NEXT:   .byte 2
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short 31
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 112
define i64 @property_access3() nounwind ssp uwtable {
entry:
  %obj = alloca i64, align 8
  %f = inttoptr i64 281474417671919 to i8*
  %ret = call anyregcc i64 (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.i64(i64 3, i32 40, i8* %f, i32 0, i64* %obj)
  ret i64 %ret
}

; anyreg_test1
; CHECK:  .long   .L{{.*}}-.L[[anyreg_test1_BEGIN]]
; CHECK-NEXT:   .short 0
; 14 locations
; CHECK-NEXT:   .short  14
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 1: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 2: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 3: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 4: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 5: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 6: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 7: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 8: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 9: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 10: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 11: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 12: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 13: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
define i64 @anyreg_test1(i8* %a1, i8* %a2, i8* %a3, i8* %a4, i8* %a5, i8* %a6, i8* %a7, i8* %a8, i8* %a9, i8* %a10, i8* %a11, i8* %a12, i8* %a13) nounwind ssp uwtable {
entry:
  %f = inttoptr i64 281474417671919 to i8*
  %ret = call anyregcc i64 (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.i64(i64 4, i32 40, i8* %f, i32 13, i8* %a1, i8* %a2, i8* %a3, i8* %a4, i8* %a5, i8* %a6, i8* %a7, i8* %a8, i8* %a9, i8* %a10, i8* %a11, i8* %a12, i8* %a13)
  ret i64 %ret
}

; anyreg_test2
; CHECK:  .long   .L{{.*}}-.L[[anyreg_test2_BEGIN]]
; CHECK-NEXT:   .short 0
; 14 locations
; CHECK-NEXT:   .short  14
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 1: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 2: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 3: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 4: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 5: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 6: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 7: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 8: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 9: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 10: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 11: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 12: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 13: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
define i64 @anyreg_test2(i8* %a1, i8* %a2, i8* %a3, i8* %a4, i8* %a5, i8* %a6, i8* %a7, i8* %a8, i8* %a9, i8* %a10, i8* %a11, i8* %a12, i8* %a13) nounwind ssp uwtable {
entry:
  %f = inttoptr i64 281474417671919 to i8*
  %ret = call anyregcc i64 (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.i64(i64 5, i32 40, i8* %f, i32 8, i8* %a1, i8* %a2, i8* %a3, i8* %a4, i8* %a5, i8* %a6, i8* %a7, i8* %a8, i8* %a9, i8* %a10, i8* %a11, i8* %a12, i8* %a13)
  ret i64 %ret
}

; Test spilling the return value of an anyregcc call.
;
; <rdar://problem/15432754> [JS] Assertion: "Folded a def to a non-store!"
;
; CHECK: .long .L{{.*}}-.L[[patchpoint_spilldef_BEGIN]]
; CHECK-NEXT: .short 0
; CHECK-NEXT: .short 3
; Loc 0: Register (some register that will be spilled to the stack)
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .short 8
; CHECK-NEXT: .short {{[0-9]+}}
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long  0
; Loc 1: Register
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .short 8
; CHECK-NEXT: .short {{[0-9]+}}
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long  0
; Loc 1: Register
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .short 8
; CHECK-NEXT: .short {{[0-9]+}}
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long  0
define i64 @patchpoint_spilldef(i64 %p1, i64 %p2, i64 %p3, i64 %p4) {
entry:
  %result = tail call anyregcc i64 (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.i64(i64 12, i32 40, i8* inttoptr (i64 0 to i8*), i32 2, i64 %p1, i64 %p2)
  tail call void asm sideeffect "nop", "~{r0},~{r3},~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12},~{r14},~{r15},~{r16},~{r17
},~{r18},~{r19},~{r20},~{r21},~{r22},~{r23},~{r24},~{r25},~{r26},~{r27},~{r28},~{r29},~{r30},~{r31}"() nounwind
  ret i64 %result
}

; Test spilling the arguments of an anyregcc call.
;
; <rdar://problem/15487687> [JS] AnyRegCC argument ends up being spilled
;
; CHECK: .long .L{{.*}}-.L[[patchpoint_spillargs_BEGIN]]
; CHECK-NEXT: .short 0
; CHECK-NEXT: .short 5
; Loc 0: Return a register
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .short 8
; CHECK-NEXT: .short {{[0-9]+}}
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long  0
; Loc 1: Arg0 in a Register
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .short 8
; CHECK-NEXT: .short {{[0-9]+}}
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long  0
; Loc 2: Arg1 in a Register
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .short 8
; CHECK-NEXT: .short {{[0-9]+}}
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long  0
; Loc 3: Arg2 spilled to FP -96
; CHECK-NEXT: .byte  3
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .short 8
; CHECK-NEXT: .short 31
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long 128
; Loc 4: Arg3 spilled to FP - 88
; CHECK-NEXT: .byte  3
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .short 8
; CHECK-NEXT: .short 31
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long 136
define i64 @patchpoint_spillargs(i64 %p1, i64 %p2, i64 %p3, i64 %p4) {
entry:
  tail call void asm sideeffect "nop", "~{r0},~{r3},~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12},~{r14},~{r15},~{r16},~{r17
},~{r18},~{r19},~{r20},~{r21},~{r22},~{r23},~{r24},~{r25},~{r26},~{r27},~{r28},~{r29},~{r30},~{r31}"() nounwind
  %result = tail call anyregcc i64 (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.i64(i64 13, i32 40, i8* inttoptr (i64 0 to i8*), i32 2, i64 %p1, i64 %p2, i64 %p3, i64 %p4)
  ret i64 %result
}

declare void @llvm.experimental.patchpoint.void(i64, i32, i8*, i32, ...)
declare i64 @llvm.experimental.patchpoint.i64(i64, i32, i8*, i32, ...)
