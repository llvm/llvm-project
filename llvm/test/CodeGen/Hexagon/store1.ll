; RUN: llc -march=hexagon < %s | FileCheck %s
;
; Check that the immediate form for the store instructions are generated.
;
; CHECK: memw(r{{[0-9]+}}+#156) = #0
; CHECK: memw(r{{[0-9]+}}+#160) = ##g0+144
; CHECK: memw(r{{[0-9]+}}+#172) = ##f3

%s.0 = type { [156 x i8], ptr, ptr, i8, ptr, ptr, i8 }

@g0 = common global %s.0 zeroinitializer, align 4

; Function Attrs: nounwind
define void @f0(ptr %a0) #0 {
b0:
  %v0 = getelementptr inbounds %s.0, ptr %a0, i32 0, i32 1
  store ptr null, ptr %v0, align 4
  ret void
}

; Function Attrs: nounwind
define void @f1(ptr %a0) #0 {
b0:
  %v0 = getelementptr inbounds %s.0, ptr %a0, i32 0, i32 2
  store ptr getelementptr inbounds (%s.0, ptr @g0, i32 0, i32 0, i32 144), ptr %v0, align 4
  ret void
}

; Function Attrs: nounwind
define void @f2(ptr %a0) #0 {
b0:
  %v0 = getelementptr inbounds %s.0, ptr %a0, i32 0, i32 5
  store ptr @f3, ptr %v0, align 4
  ret void
}

declare void @f3(ptr)

attributes #0 = { nounwind }
