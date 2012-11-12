; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; PR9857

define void @f(i8** nocapture %ptr1) {
; CHECK: define void @f
entry:
  br label %here.i

here.i:
  store i8* blockaddress(@doit, %here), i8** %ptr1, align 8
; CHECK: blockaddress(@doit, %here)
  br label %doit.exit

doit.exit:
  ret void
}

define void @doit(i8** nocapture %pptr) {
; CHECK: define void @doit
entry:
  br label %here

here:
  store i8* blockaddress(@doit, %here), i8** %pptr, align 8
; CHECK: blockaddress(@doit, %here)
  br label %end

end:
  ret void
}

; PR13895
define void @doitagain(i8** nocapture %pptr) {
; CHECK: define void @doitagain
entry:
  br label %here

here:
  store i8* blockaddress(@doit, %here), i8** %pptr, align 8
; CHECK: blockaddress(@doit, %here)
  br label %end

end:
  ret void
}
