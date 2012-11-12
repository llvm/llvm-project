; RUN: opt -basicaa -inline -S -scalarrepl -gvn -instcombine %s | FileCheck %s
; PR5009

; CHECK: define i32 @main() 
; CHECK-NEXT: entry:
; CHECK-NEXT:  call void @exit(i32 38) 

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

%struct.cont_t = type { void (i8*, i32)*, i8* }
%struct.foo_sf_t = type { %struct.cont_t*, i32 }

define i32 @main() nounwind ssp {
entry:
  %cont = alloca %struct.cont_t, align 8          ; <%struct.cont_t*> [#uses=4]
  %tmp = getelementptr inbounds %struct.cont_t* %cont, i32 0, i32 0 ; <void (i8*, i32)**> [#uses=1]
  %tmp1 = getelementptr inbounds %struct.cont_t* %cont, i32 0, i32 0 ; <void (i8*, i32)**> [#uses=2]
  store void (i8*, i32)* bitcast (void (%struct.cont_t*, i32)* @quit to void (i8*, i32)*), void (i8*, i32)** %tmp1
  %tmp2 = load void (i8*, i32)** %tmp1            ; <void (i8*, i32)*> [#uses=1]
  store void (i8*, i32)* %tmp2, void (i8*, i32)** %tmp
  %tmp3 = getelementptr inbounds %struct.cont_t* %cont, i32 0, i32 1 ; <i8**> [#uses=1]
  store i8* null, i8** %tmp3
  call void @foo(%struct.cont_t* %cont)
  ret i32 0
}

define internal void @quit(%struct.cont_t* %cont, i32 %rcode) nounwind ssp {
entry:
  call void @exit(i32 %rcode) noreturn
  unreachable
}

define internal void @foo(%struct.cont_t* %c) nounwind ssp {
entry:
  %sf = alloca %struct.foo_sf_t, align 8          ; <%struct.foo_sf_t*> [#uses=3]
  %next = alloca %struct.cont_t, align 8          ; <%struct.cont_t*> [#uses=3]
  %tmp = getelementptr inbounds %struct.foo_sf_t* %sf, i32 0, i32 0 ; <%struct.cont_t**> [#uses=1]
  store %struct.cont_t* %c, %struct.cont_t** %tmp
  %tmp2 = getelementptr inbounds %struct.foo_sf_t* %sf, i32 0, i32 1 ; <i32*> [#uses=1]
  store i32 2, i32* %tmp2
  %tmp4 = getelementptr inbounds %struct.cont_t* %next, i32 0, i32 0 ; <void (i8*, i32)**> [#uses=1]
  store void (i8*, i32)* bitcast (void (%struct.foo_sf_t*, i32)* @foo2 to void (i8*, i32)*), void (i8*, i32)** %tmp4
  %tmp5 = getelementptr inbounds %struct.cont_t* %next, i32 0, i32 1 ; <i8**> [#uses=1]
  %conv = bitcast %struct.foo_sf_t* %sf to i8*    ; <i8*> [#uses=1]
  store i8* %conv, i8** %tmp5
  call void @bar(%struct.cont_t* %next, i32 14)
  ret void
}

define internal void @foo2(%struct.foo_sf_t* %sf, i32 %y) nounwind ssp {
entry:
  %tmp1 = getelementptr inbounds %struct.foo_sf_t* %sf, i32 0, i32 0 ; <%struct.cont_t**> [#uses=1]
  %tmp2 = load %struct.cont_t** %tmp1             ; <%struct.cont_t*> [#uses=1]
  %tmp3 = getelementptr inbounds %struct.cont_t* %tmp2, i32 0, i32 0 ; <void (i8*, i32)**> [#uses=1]
  %tmp4 = load void (i8*, i32)** %tmp3            ; <void (i8*, i32)*> [#uses=1]
  %tmp6 = getelementptr inbounds %struct.foo_sf_t* %sf, i32 0, i32 0 ; <%struct.cont_t**> [#uses=1]
  %tmp7 = load %struct.cont_t** %tmp6             ; <%struct.cont_t*> [#uses=1]
  %conv = bitcast %struct.cont_t* %tmp7 to i8*    ; <i8*> [#uses=1]
  %tmp9 = getelementptr inbounds %struct.foo_sf_t* %sf, i32 0, i32 1 ; <i32*> [#uses=1]
  %tmp10 = load i32* %tmp9                        ; <i32> [#uses=1]
  %mul = mul i32 %tmp10, %y                       ; <i32> [#uses=1]
  call void %tmp4(i8* %conv, i32 %mul)
  ret void
}

define internal void @bar(%struct.cont_t* %c, i32 %y) nounwind ssp {
entry:
  %tmp1 = getelementptr inbounds %struct.cont_t* %c, i32 0, i32 0 ; <void (i8*, i32)**> [#uses=1]
  %tmp2 = load void (i8*, i32)** %tmp1            ; <void (i8*, i32)*> [#uses=1]
  %tmp4 = getelementptr inbounds %struct.cont_t* %c, i32 0, i32 1 ; <i8**> [#uses=1]
  %tmp5 = load i8** %tmp4                         ; <i8*> [#uses=1]
  %add = add nsw i32 %y, 5                        ; <i32> [#uses=1]
  call void %tmp2(i8* %tmp5, i32 %add)
  ret void
}

declare void @exit(i32) noreturn

