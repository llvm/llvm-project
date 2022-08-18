; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -enable-patchpoint-liveness=false | FileCheck %s

; CHECK-LABEL:  .section  __LLVM_STACKMAPS,__llvm_stackmaps
; CHECK-NEXT:   __LLVM_StackMaps:

; Header
; CHECK-NEXT:   .byte 3
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 0

; NumFunctions
; CHECK-NEXT:   .long 1
; NumConstants
; CHECK-NEXT:   .long 0
; NumRecords
; CHECK-NEXT:   .long 1

; StackSizeRecord[NumFunctions]
;   StackSizeRecord[0]
;     CHECK-NEXT:   .quad _main
;     CHECK-NEXT:   .quad 24
;     CHECK-NEXT:   .quad 1

; Constants[NumConstants] (empty)

; StkMapRecord[NumRecords]
;   StkMapRecord[0]
;     CHECK-NEXT:	.quad 0
;     CHECK-NEXT:   .long {{.*}}
;     CHECK-NEXT:   .short {{.*}}
;     NumLiveVars
;     CHECK-NEXT:   .short 13
;     LiveVars[NumLiveVars]
;       LiveVars[0]
;       CHECK-NEXT: .byte   1
;       Locations[0]
;         CHECK-NEXT: .byte   1
;         CHECK-NEXT: .byte   0
;         CHECK-NEXT: .short  1
;         CHECK-NEXT: .short  {{.*}}
;         CHECK-NEXT: .short  0
;         CHECK-NEXT: .long   0
;       LiveVars[1]
;         CHECK-NEXT: .byte   1
;         Locations[0]
;           CHECK-NEXT: .byte   4
;           CHECK-NEXT: .byte   0
;           CHECK-NEXT: .short  8
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .long   22
;       LiveVars[2]
;         CHECK-NEXT: .byte   1
;         Locations[0]
;           CHECK-NEXT: .byte   1
;           CHECK-NEXT: .byte   0
;           CHECK-NEXT: .short  1
;           CHECK-NEXT: .short  {{.*}}
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .long   0
;       LiveVars[3]
;         CHECK-NEXT: .byte   1
;         Locations[0]
;           CHECK-NEXT: .byte   1
;           CHECK-NEXT: .byte   0
;           CHECK-NEXT: .short  16
;           CHECK-NEXT: .short  {{.*}}
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .long   0
;       LiveVars[4]
;         CHECK-NEXT: .byte   1
;         Locations[0]
;           CHECK-NEXT: .byte   1
;           CHECK-NEXT: .byte   0
;           CHECK-NEXT: .short  16
;           CHECK-NEXT: .short  {{.*}}
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .long   0
;       LiveVars[5]
;         CHECK-NEXT: .byte   1
;         Locations[0]
;           CHECK-NEXT: .byte   4
;           CHECK-NEXT: .byte   0
;           CHECK-NEXT: .short  8
;           CHECK-NEXT: .short  {{.*}}
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .long   66
;       LiveVars[6]
;         CHECK-NEXT: .byte   1
;         Locations[0]
;           CHECK-NEXT: .byte   1
;           CHECK-NEXT: .byte   0
;           CHECK-NEXT: .short  4
;           CHECK-NEXT: .short  {{.*}}
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .long   0
;       LiveVars[7]
;         CHECK-NEXT: .byte   2
;         Locations[0]
;           CHECK-NEXT: .byte   4
;           CHECK-NEXT: .byte   0
;           CHECK-NEXT: .short  8
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .long   0
;         Locations[1]
;           CHECK-NEXT: .byte   4
;           CHECK-NEXT: .byte   0
;           CHECK-NEXT: .short  8
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .long   0
;       LiveVars[8]
;         CHECK-NEXT: .byte   2
;         Locations[0]
;           CHECK-NEXT: .byte   1
;           CHECK-NEXT: .byte   0
;           CHECK-NEXT: .short  4
;           CHECK-NEXT: .short  {{.*}}
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .long   0
;         Locations[1]
;           CHECK-NEXT: .byte   4
;           CHECK-NEXT: .byte   0
;           CHECK-NEXT: .short  8
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .long   0
;       LiveVars[9]
;         CHECK-NEXT: .byte   3
;         Locations[0]
;           CHECK-NEXT: .byte   4
;           CHECK-NEXT: .byte   0
;           CHECK-NEXT: .short  8
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .long   0
;         Locations[1]
;           CHECK-NEXT: .byte   4
;           CHECK-NEXT: .byte   0
;           CHECK-NEXT: .short  8
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .long   0
;         Locations[2]
;           CHECK-NEXT: .byte   4
;           CHECK-NEXT: .byte   0
;           CHECK-NEXT: .short  8
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .long   0
;       LiveVars[10]
;         CHECK-NEXT: .byte   3
;         Locations[0]
;           CHECK-NEXT: .byte   1
;           CHECK-NEXT: .byte   0
;           CHECK-NEXT: .short  1
;           CHECK-NEXT: .short  {{.*}}
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .long   0
;         Locations[1]
;           CHECK-NEXT: .byte   4
;           CHECK-NEXT: .byte   0
;           CHECK-NEXT: .short  8
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .long   0
;         Locations[2]
;           CHECK-NEXT: .byte   4
;           CHECK-NEXT: .byte   0
;           CHECK-NEXT: .short  8
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .long   0
;       LiveVars[11]
;         CHECK-NEXT: .byte   4
;         Locations[0]
;           CHECK-NEXT: .byte   4
;           CHECK-NEXT: .byte   0
;           CHECK-NEXT: .short  8
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .long   0
;         Locations[1]
;           CHECK-NEXT: .byte   4
;           CHECK-NEXT: .byte   0
;           CHECK-NEXT: .short  8
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .long   1
;         Locations[1]
;           CHECK-NEXT: .byte   4
;           CHECK-NEXT: .byte   0
;           CHECK-NEXT: .short  8
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .long   2
;         Locations[1]
;           CHECK-NEXT: .byte   4
;           CHECK-NEXT: .byte   0
;           CHECK-NEXT: .short  8
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .long   3
;       LiveVars[12]
;         CHECK-NEXT: .byte   4
;         Locations[0]
;           CHECK-NEXT: .byte   1
;           CHECK-NEXT: .byte   0
;           CHECK-NEXT: .short  4
;           CHECK-NEXT: .short  {{.*}}
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .long   0
;         Locations[1]
;           CHECK-NEXT: .byte   4
;           CHECK-NEXT: .byte   0
;           CHECK-NEXT: .short  8
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .long   0
;         Locations[1]
;           CHECK-NEXT: .byte   4
;           CHECK-NEXT: .byte   0
;           CHECK-NEXT: .short  8
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .long   0
;         Locations[1]
;           CHECK-NEXT: .byte   4
;           CHECK-NEXT: .byte   0
;           CHECK-NEXT: .short  8
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .short  0
;           CHECK-NEXT: .long   0

@p32 = external global i8 addrspace(270)*

%struct1 = type {i32, i64}
%struct2 = type {i1, i1, i1}

declare void @llvm.experimental.stackmap(i64, i32, ...)

define dso_local i32 @main(i32 %argc, i8** %argv) {
entry:
  %i1reg = icmp eq i32 %argc, 5
  %i7reg = zext i1 %i1reg to i7
  %halfreg = sitofp i32 %argc to half
  %ptr32 = load i8 addrspace(270)*, i8 addrspace(270)** @p32
  %structreg1 = insertvalue %struct1 zeroinitializer, i32 %argc, 0
  %structreg2 = insertvalue %struct2 zeroinitializer, i1 %i1reg, 0
  %arrayreg = insertvalue [4 x i32] zeroinitializer, i32 %argc, 0
  call void (i64, i32, ...) @llvm.experimental.stackmap(
    i64 0,
    i32 0,
    i1 %i1reg,
    i7 22,
    i7 %i7reg,
    half 1.0,
    half %halfreg,
    i128 66,
    ; FIXME: fix and test vectors. At the moment even legally sized vectors
    ; are broken:
    ; https://github.com/llvm/llvm-project/issues/55613
    ;
    ; FIXME: test non-constant i128 once these are fixed:
    ;  - https://github.com/llvm/llvm-project/issues/26431
    ;  - https://github.com/llvm/llvm-project/issues/55957
    i8 addrspace(270)* %ptr32,
    ; FIXME: The stackmap record generated for structs is incorrect:
    ;  - https://github.com/llvm/llvm-project/issues/55649
    ;  - https://github.com/llvm/llvm-project/issues/55957
    %struct1 zeroinitializer,
    %struct1 %structreg1,
    %struct2 zeroinitializer,
    %struct2 %structreg2,
    [4 x i32] [i32 0, i32 1, i32 2, i32 3],
    [4 x i32] %arrayreg)
  ret i32 0
}
