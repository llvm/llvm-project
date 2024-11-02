; RUN: opt -passes=consthoist -S < %s | FileCheck %s
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

; Indices for GEPs should not be hoisted.
define i32 @test1(%T* %P) nounwind {
; CHECK-LABEL:  @test1
; CHECK:        %addr1 = getelementptr %T, %T* %P, i32 256, i32 256
; CHECK:        %addr2 = getelementptr %T, %T* %P, i32 256, i32 256
  %addr1 = getelementptr %T, %T* %P, i32 256, i32 256
  %tmp1 = load i32, i32* %addr1
  %addr2 = getelementptr %T, %T* %P, i32 256, i32 256
  %tmp2 = load i32, i32* %addr2
  %tmp4 = add i32 %tmp1, %tmp2
  ret i32 %tmp4
}

