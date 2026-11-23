; RUN: opt -passes='print<ir2vec>' -disable-output -ir2vec-vocab-path=%S/Inputs/dummy_3D_nonzero_opc_vocab.json %s 2>&1 | FileCheck %s
; RUN: opt -passes='print<ir2vec>' -disable-output -ir2vec-vocab-path=%S/Inputs/dummy_3D_nonzero_arg_vocab.json %s 2>&1 | FileCheck %s -check-prefix=ARG

define dso_local i32 @abc(i32 noundef %a, i32 noundef %b) #0 {
entry:
  %retval = alloca i32, align 4
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  store i32 %b, ptr %b.addr, align 4
  %0 = load i32, ptr %a.addr, align 4
  %1 = load i32, ptr %b.addr, align 4
  %cmp = icmp sgt i32 %0, %1
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %2 = load i32, ptr %b.addr, align 4
  store i32 %2, ptr %retval, align 4
  br label %return

if.else:                                          ; preds = %entry
  %3 = load i32, ptr %a.addr, align 4
  store i32 %3, ptr %retval, align 4
  br label %return

return:                                           ; preds = %if.else, %if.then
  %4 = load i32, ptr %retval, align 4
  ret i32 %4
}

; CHECK: Basic block vectors:
; CHECK-NEXT: Basic block: entry:
; CHECK-NEXT: [ 816.20  825.20  834.20 ]
; CHECK-NEXT: Basic block: if.then:
; CHECK-NEXT: [ 195.00  198.00  201.00 ]
; CHECK-NEXT: Basic block: if.else:
; CHECK-NEXT: [ 195.00  198.00  201.00 ]
; CHECK-NEXT: Basic block: return:
; CHECK-NEXT: [ 95.00  97.00  99.00 ]

; ARG: Basic block vectors:
; ARG-NEXT: Basic block: entry:
; ARG-NEXT: [ 69.60  72.60  75.60 ]
; ARG-NEXT: Basic block: if.then:
; ARG-NEXT: [ 21.80  22.60  23.40 ]
; ARG-NEXT: Basic block: if.else:
; ARG-NEXT: [ 21.80  22.60  23.40 ]
; ARG-NEXT: Basic block: return:
; ARG-NEXT: [ 2.80  3.20  3.60 ]
