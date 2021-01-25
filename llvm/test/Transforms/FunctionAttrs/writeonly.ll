; RUN: opt < %s -function-attrs         -S | FileCheck %s
; RUN: opt < %s -passes=function-attrs -S | FileCheck %s

; CHECK: define void @nouses-argworn-funrn(i32* nocapture readnone %.aaa) #0 {
define void @nouses-argworn-funrn(i32* writeonly %.aaa) {
nouses-argworn-funrn_entry:
  ret void
}

; CHECK: define void @nouses-argworn-funro(i32* nocapture readnone %.aaa, i32* nocapture readonly %.bbb) #1 {
define void @nouses-argworn-funro(i32* writeonly %.aaa, i32* %.bbb) {
nouses-argworn-funro_entry:
  %val = load i32 , i32* %.bbb
  ret void
}

%_type_of_d-ccc = type <{ i8*, i8, i8, i8, i8 }>

@d-ccc = internal global %_type_of_d-ccc <{ i8* null, i8 1, i8 13, i8 0, i8 -127 }>, align 8

; CHECK: define void @nouses-argworn-funwo(i32* nocapture readnone %.aaa) #2 {
define void @nouses-argworn-funwo(i32* writeonly %.aaa) {
nouses-argworn-funwo_entry:
  store i8 0, i8* getelementptr inbounds (%_type_of_d-ccc, %_type_of_d-ccc* @d-ccc, i32 0, i32 3)
  ret void
}

; CHECK: attributes #0 = { {{.*}} readnone {{.*}} }
; CHECK: attributes #1 = { {{.*}} readonly {{.*}} }
; CHECK: attributes #2 = { {{.*}} writeonly }
