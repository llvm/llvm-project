; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Stackmap Header: no constants - 6 callsites
; CHECK:        .section .llvm_stackmaps
; CHECK-NEXT:   __LLVM_StackMaps:
; Header
; CHECK-NEXT:   .byte 3
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 0
; Num Functions
; CHECK-NEXT:   .long 8
; Num Constants
; CHECK-NEXT:   .long 0
; Num Callsites
; CHECK-NEXT:   .long 8

; Functions and stack size
; CHECK-NEXT:   .quad test
; CHECK-NEXT:   .quad 160
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad property_access1
; CHECK-NEXT:   .quad 160
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad property_access2
; CHECK-NEXT:   .quad 168
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad property_access3
; CHECK-NEXT:   .quad 168
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad anyreg_test1
; CHECK-NEXT:   .quad 160
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad anyreg_test2
; CHECK-NEXT:   .quad 160
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad patchpoint_spilldef
; CHECK-NEXT:   .quad 168
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad patchpoint_spillargs
; CHECK-NEXT:   .quad 192
; CHECK-NEXT:   .quad 1

; No constants

; Callsites
; test
; CHECK:        .long   .L{{.*}}-test
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
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 3
define i64 @test() nounwind ssp uwtable {
entry:
  call anyregcc void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 0, i32 14, ptr null, i32 2, i32 1, i32 2, i64 3)
  ret i64 0
}

; property access 1 - %obj is an anyreg call argument and should therefore be in a register
; CHECK:        .long   .L{{.*}}-property_access1
; CHECK-NEXT:   .short  0
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
define i64 @property_access1(ptr %obj) nounwind ssp uwtable {
entry:
  %f = inttoptr i64 12297829382473034410 to ptr
  %ret = call anyregcc i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 1, i32 14, ptr %f, i32 1, ptr %obj)
  ret i64 %ret
}

; property access 2 - %obj is an anyreg call argument and should therefore be in a register
; CHECK:        .long   .L{{.*}}-property_access2
; CHECK-NEXT:   .short  0
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
  %f = inttoptr i64 12297829382473034410 to ptr
  %ret = call anyregcc i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 2, i32 14, ptr %f, i32 1, ptr %obj)
  ret i64 %ret
}

; property access 3 - %obj is a frame index
; CHECK:        .long   .L{{.*}}-property_access3
; CHECK-NEXT:   .short  0
; 2 locations
; CHECK-NEXT:   .short  2
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 1: Direct %r15 + 160
; CHECK-NEXT:   .byte 2
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short 15
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long  160
define i64 @property_access3() nounwind ssp uwtable {
entry:
  %obj = alloca i64, align 8
  %f = inttoptr i64 12297829382473034410 to ptr
  %ret = call anyregcc i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 3, i32 14, ptr %f, i32 0, ptr %obj)
  ret i64 %ret
}

; anyreg_test1
; CHECK:        .long   .L{{.*}}-anyreg_test1
; CHECK-NEXT:   .short  0
; 13 locations
; CHECK-NEXT:   .short  13
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
define i64 @anyreg_test1(ptr %a1, ptr %a2, ptr %a3, ptr %a4, ptr %a5, ptr %a6, ptr %a7, ptr %a8, ptr %a9, ptr %a10, ptr %a11, ptr %a12) nounwind ssp uwtable {
entry:
  %f = inttoptr i64 12297829382473034410 to ptr
  %ret = call anyregcc i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 4, i32 14, ptr %f, i32 12, ptr %a1, ptr %a2, ptr %a3, ptr %a4, ptr %a5, ptr %a6, ptr %a7, ptr %a8, ptr %a9, ptr %a10, ptr %a11, ptr %a12)
  ret i64 %ret
}

; anyreg_test2
; CHECK:        .long   .L{{.*}}-anyreg_test2
; CHECK-NEXT:   .short  0
; 13 locations
; CHECK-NEXT:   .short  13
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
define i64 @anyreg_test2(ptr %a1, ptr %a2, ptr %a3, ptr %a4, ptr %a5, ptr %a6, ptr %a7, ptr %a8, ptr %a9, ptr %a10, ptr %a11, ptr %a12) nounwind ssp uwtable {
entry:
  %f = inttoptr i64 12297829382473034410 to ptr
  %ret = call anyregcc i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 5, i32 14, ptr %f, i32 8, ptr %a1, ptr %a2, ptr %a3, ptr %a4, ptr %a5, ptr %a6, ptr %a7, ptr %a8, ptr %a9, ptr %a10, ptr %a11, ptr %a12)
  ret i64 %ret
}

; Test spilling the return value of an anyregcc call.
;
; <rdar://problem/15432754> [JS] Assertion: "Folded a def to a non-store!"
;
; CHECK:      .long  .L{{.*}}-patchpoint_spilldef
; CHECK-NEXT: .short 0
; CHECK-NEXT: .short 3
; Loc 0: Register (some register that will be spilled to the stack)
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte  0
; CHECK-NEXT: .short  8
; CHECK-NEXT: .short {{[0-9]+}}
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long  0
; Loc 1: Register %r2
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte  0
; CHECK-NEXT: .short  8
; CHECK-NEXT: .short 2
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long  0
; Loc 1: Register %r3
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte  0
; CHECK-NEXT: .short  8
; CHECK-NEXT: .short 3
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long  0
define i64 @patchpoint_spilldef(i64 %p1, i64 %p2, i64 %p3, i64 %p4) {
entry:
  %result = tail call anyregcc i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 12, i32 14, ptr inttoptr (i64 0 to ptr), i32 2, i64 %p1, i64 %p2)
  tail call void asm sideeffect "nopr %r0", "~{r0},~{r1},~{r2},~{r3},~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14}"() nounwind
  ret i64 %result
}

; Test spilling the arguments of an anyregcc call.
;
; <rdar://problem/15487687> [JS] AnyRegCC argument ends up being spilled
;
; CHECK:      .long .L{{.*}}-patchpoint_spillargs
; CHECK-NEXT: .short 0
; CHECK-NEXT: .short 5
; Loc 0: Return a register
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte  0
; CHECK-NEXT: .short  8
; CHECK-NEXT: .short {{[0-9]+}}
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long  0
; Loc 1: Arg0 in a Register
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte  0
; CHECK-NEXT: .short  8
; CHECK-NEXT: .short {{[0-9]+}}
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long  0
; Loc 2: Arg1 in a Register
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte  0
; CHECK-NEXT: .short  8
; CHECK-NEXT: .short {{[0-9]+}}
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long  0
; Loc 3: Arg2 spilled to %r15 +
; CHECK-NEXT: .byte  3
; CHECK-NEXT: .byte  0
; CHECK-NEXT: .short  8
; CHECK-NEXT: .short 15
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long
; Loc 4: Arg3 spilled to %r15 +
; CHECK-NEXT: .byte  3
; CHECK-NEXT: .byte  0
; CHECK-NEXT: .short  8
; CHECK-NEXT: .short 15
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long
define i64 @patchpoint_spillargs(i64 %p1, i64 %p2, i64 %p3, i64 %p4) {
entry:
  tail call void asm sideeffect "nopr %r0", "~{r0},~{r1},~{r2},~{r3},~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14}"() nounwind
  %result = tail call anyregcc i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 13, i32 14, ptr inttoptr (i64 0 to ptr), i32 2, i64 %p1, i64 %p2, i64 %p3, i64 %p4)
  ret i64 %result
}

declare void @llvm.experimental.patchpoint.void(i64, i32, ptr, i32, ...)
declare i64 @llvm.experimental.patchpoint.i64(i64, i32, ptr, i32, ...)
