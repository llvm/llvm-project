; RUN: llc -march=hexagon < %s | FileCheck %s

; Make sure that the addressing mode optimization does not propagate
; an add instruction where the base register would have a different
; reaching def.

; CHECK-LABEL: f0.1:
; CHECK-LABEL: %b0
; CHECK:         r17 = add(r{{[0-9]+}},#8)
; CHECK-LABEL: %b1
; CHECK:         r16 = r0
; CHECK-LABEL: %b2
; CHECK:         memd(r17+#0)

target triple = "hexagon"

%s.0 = type { i8, i8, %s.1, i32 }
%s.1 = type { %s.2, [128 x i8] }
%s.2 = type { i8, i8, i64, %s.3 }
%s.3 = type { i8 }

define void @f0.1() local_unnamed_addr #0 align 2 {
b0:
  %v0 = alloca %s.0, align 8
  %v1 = getelementptr inbounds %s.0, %s.0* %v0, i32 0, i32 1
  store i8 4, i8* %v1, align 1
  %v2 = call signext i8 @f1.2(%s.3* undef) #0
  %v3 = getelementptr inbounds %s.0, %s.0* %v0, i32 0, i32 2, i32 0, i32 0
  %v4 = getelementptr inbounds %s.0, %s.0* %v0, i32 0, i32 2, i32 0, i32 3, i32 0
  store i8 -1, i8* %v4, align 8
  br i1 undef, label %b1, label %b2

b1:                                               ; preds = %b0
  %v5 = call dereferenceable(12) %s.3* @f2.3(%s.3* nonnull undef, %s.3* nonnull dereferenceable(80) undef) #0
  %v6 = call signext i8 @f1.2(%s.3* undef) #0
  %v7 = call dereferenceable(12) %s.3* @f3(%s.3* nonnull %v5, i16 signext undef) #0
  br label %b2

b2:                                               ; preds = %b1, %b0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 undef, i8* align 8 %v3, i32 48, i1 false)
  ret void
}

declare signext i8 @f1.2(%s.3*) #0
declare dereferenceable(12) %s.3* @f2.3(%s.3*, %s.3* dereferenceable(80)) #0
declare dereferenceable(12) %s.3* @f3(%s.3*, i16 signext) #0
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture writeonly, i8* nocapture readonly, i32, i1) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="-long-calls" }
attributes #1 = { argmemonly nounwind }
