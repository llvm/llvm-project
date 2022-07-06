; RUN: llc --mtriple=loongarch64 < %s | FileCheck %s

%struct.key_t = type { i32, [16 x i8] }

;; FIXME: prologue and epilogue insertion must be implemented to complete this
;; test

define i32 @test() nounwind {
; CHECK-LABEL: test:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st.d $ra, $sp, 24 # 8-byte Folded Spill
; CHECK-NEXT:    st.w $zero, $sp, 16
; CHECK-NEXT:    st.d $zero, $sp, 8
; CHECK-NEXT:    st.d $zero, $sp, 0
; CHECK-NEXT:    addi.d $a0, $sp, 0
; CHECK-NEXT:    ori $a0, $a0, 4
; CHECK-NEXT:    bl test1
; CHECK-NEXT:    move $a0, $zero
; CHECK-NEXT:    ld.d $ra, $sp, 24 # 8-byte Folded Reload
; CHECK-NEXT:    jirl $zero, $ra, 0
  %key = alloca %struct.key_t, align 4
  call void @llvm.memset.p0i8.i64(ptr %key, i8 0, i64 20, i1 false)
  %1 = getelementptr inbounds %struct.key_t, ptr %key, i64 0, i32 1, i64 0
  call void @test1(ptr %1)
  ret i32 0
}

declare void @llvm.memset.p0i8.i64(ptr, i8, i64, i1)

declare void @test1(ptr)
