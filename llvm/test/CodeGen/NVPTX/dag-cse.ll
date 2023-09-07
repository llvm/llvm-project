; RUN: llc < %s -march=nvptx64 | FileCheck %s

%st = type { i8, i8, i16 }

@a = internal addrspace(1) global %st zeroinitializer, align 8
@b = internal addrspace(1) global i32 0, align 8
@c = internal addrspace(1) global i32 0, align 8

; Verify that loads with different memory types are not subject to CSE
; once they are promoted to the same type.
;
; CHECK: ld.global.v2.u8  {%[[B1:rs[0-9]+]], %[[B2:rs[0-9]+]]}, [a];
; CHECK: st.global.v2.u8  [b], {%[[B1]], %[[B2]]};
;
; CHECK: ld.global.u32 %[[C:r[0-9]+]], [a];
; CHECK: st.global.u32 [c], %[[C]];

define void @test1() #0 {
  %1 = load <2 x i8>, ptr addrspace(1) @a, align 8
  store <2 x i8> %1, ptr addrspace(1) @b, align 8
  %2 = load <2 x i16>, ptr addrspace(1) @a, align 8
  store <2 x i16> %2, ptr addrspace(1) @c, align 8
  ret void
}
