; RUN: llc -O1 -mtriple=aarch64-none-linux-gnu -mattr=+mops -o - %s  | FileCheck %s

define void @call_memset_intrinsic() #0 {
; CHECK-LABEL: call_memset_intrinsic:
; CHECK:       // %bb.0: // %entry
; CHECK:         setp [x{{[0-9]+}}]!, x{{[0-9]+}}!, x{{[0-9]+}}
; CHECK-NOT:     setp [x{{[0-9]+}}]!, x[[REG:[0-9]+]]!, x[[REG]]
; CHECK-NEXT:    setm [x{{[0-9]+}}]!, x{{[0-9]+}}!, x{{[0-9]+}}
; CHECK-NOT:     setm [x{{[0-9]+}}]!, x[[REG:[0-9]+]]!, x[[REG]]
; CHECK-NEXT:    sete [x{{[0-9]+}}]!, x{{[0-9]+}}!, x{{[0-9]+}}
; CHECK-NOT:     sete [x{{[0-9]+}}]!, x[[REG:[0-9]+]]!, x[[REG]]
entry:

    %V0 = alloca [65 x i8], align 1
    call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(64) %V0, i8 64, i64 64, i1 false)
    %add.ptr = getelementptr inbounds i8, ptr %V0, i64 64
     call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(64) %add.ptr, i8 64, i64 64, i1 false)
    ret void
}

attributes #0 = { "target-cpu"="generic" "target-features"="+mops,+strict-align,+v9.3a" }
