; RUN: opt -consthoist -S < %s | FileCheck %s
target triple = "thumbv6m-none-eabi"

%T = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
i32, i32, i32, i32, i32, i32 }

; Indices for GEPs that index into a struct type should not be hoisted.
define i32 @test1(%T* %P) nounwind {
; CHECK-LABEL:  @test1
; CHECK:        %const = bitcast i32 256 to i32
; CHECK:        %addr1 = getelementptr %T, %T* %P, i32 %const, i32 256
; CHECK:        %addr2 = getelementptr %T, %T* %P, i32 %const, i32 256
; The first index into the pointer is hoisted, but the second one into the
; struct isn't.
  %addr1 = getelementptr %T, %T* %P, i32 256, i32 256
  %tmp1 = load i32, i32* %addr1
  %addr2 = getelementptr %T, %T* %P, i32 256, i32 256
  %tmp2 = load i32, i32* %addr2
  %tmp4 = add i32 %tmp1, %tmp2
  ret i32 %tmp4
}

