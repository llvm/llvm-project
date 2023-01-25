; RUN: opt -aa-pipeline=basic-aa -S -passes='default<O2>' < %s | FileCheck %s
; PR5009

; CHECK: define i32 @main() 
; CHECK-NEXT: entry:
; CHECK-NEXT:  call void @exit(i32 38) 

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

%struct.cont_t = type { ptr, ptr }
%struct.foo_sf_t = type { ptr, i32 }

define i32 @main() nounwind ssp {
entry:
  %cont = alloca %struct.cont_t, align 8          ; <ptr> [#uses=4]
  %tmp = getelementptr inbounds %struct.cont_t, ptr %cont, i32 0, i32 0 ; <ptr> [#uses=1]
  %tmp1 = getelementptr inbounds %struct.cont_t, ptr %cont, i32 0, i32 0 ; <ptr> [#uses=2]
  store ptr @quit, ptr %tmp1
  %tmp2 = load ptr, ptr %tmp1            ; <ptr> [#uses=1]
  store ptr %tmp2, ptr %tmp
  %tmp3 = getelementptr inbounds %struct.cont_t, ptr %cont, i32 0, i32 1 ; <ptr> [#uses=1]
  store ptr null, ptr %tmp3
  call void @foo(ptr %cont)
  ret i32 0
}

define internal void @quit(ptr %cont, i32 %rcode) nounwind ssp {
entry:
  call void @exit(i32 %rcode) noreturn
  unreachable
}

define internal void @foo(ptr %c) nounwind ssp {
entry:
  %sf = alloca %struct.foo_sf_t, align 8          ; <ptr> [#uses=3]
  %next = alloca %struct.cont_t, align 8          ; <ptr> [#uses=3]
  %tmp = getelementptr inbounds %struct.foo_sf_t, ptr %sf, i32 0, i32 0 ; <ptr> [#uses=1]
  store ptr %c, ptr %tmp
  %tmp2 = getelementptr inbounds %struct.foo_sf_t, ptr %sf, i32 0, i32 1 ; <ptr> [#uses=1]
  store i32 2, ptr %tmp2
  %tmp4 = getelementptr inbounds %struct.cont_t, ptr %next, i32 0, i32 0 ; <ptr> [#uses=1]
  store ptr @foo2, ptr %tmp4
  %tmp5 = getelementptr inbounds %struct.cont_t, ptr %next, i32 0, i32 1 ; <ptr> [#uses=1]
  store ptr %sf, ptr %tmp5
  call void @bar(ptr %next, i32 14)
  ret void
}

define internal void @foo2(ptr %sf, i32 %y) nounwind ssp {
entry:
  %tmp1 = getelementptr inbounds %struct.foo_sf_t, ptr %sf, i32 0, i32 0 ; <ptr> [#uses=1]
  %tmp2 = load ptr, ptr %tmp1             ; <ptr> [#uses=1]
  %tmp3 = getelementptr inbounds %struct.cont_t, ptr %tmp2, i32 0, i32 0 ; <ptr> [#uses=1]
  %tmp4 = load ptr, ptr %tmp3            ; <ptr> [#uses=1]
  %tmp6 = getelementptr inbounds %struct.foo_sf_t, ptr %sf, i32 0, i32 0 ; <ptr> [#uses=1]
  %tmp7 = load ptr, ptr %tmp6             ; <ptr> [#uses=1]
  %tmp9 = getelementptr inbounds %struct.foo_sf_t, ptr %sf, i32 0, i32 1 ; <ptr> [#uses=1]
  %tmp10 = load i32, ptr %tmp9                        ; <i32> [#uses=1]
  %mul = mul i32 %tmp10, %y                       ; <i32> [#uses=1]
  call void %tmp4(ptr %tmp7, i32 %mul)
  ret void
}

define internal void @bar(ptr %c, i32 %y) nounwind ssp {
entry:
  %tmp1 = getelementptr inbounds %struct.cont_t, ptr %c, i32 0, i32 0 ; <ptr> [#uses=1]
  %tmp2 = load ptr, ptr %tmp1            ; <ptr> [#uses=1]
  %tmp4 = getelementptr inbounds %struct.cont_t, ptr %c, i32 0, i32 1 ; <ptr> [#uses=1]
  %tmp5 = load ptr, ptr %tmp4                         ; <ptr> [#uses=1]
  %add = add nsw i32 %y, 5                        ; <i32> [#uses=1]
  call void %tmp2(ptr %tmp5, i32 %add)
  ret void
}

declare void @exit(i32) noreturn

