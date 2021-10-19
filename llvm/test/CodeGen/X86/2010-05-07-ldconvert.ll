; RUN: llc < %s -mtriple=x86_64-apple-darwin11
; PR 7087 - used to crash

define i32 @main() ssp {
entry:
  %retval = alloca i32, align 4                   ; <i32*> [#uses=2]
  %r = alloca i32, align 4                        ; <i32*> [#uses=2]
  store i32 0, i32* %retval
  %tmp = call x86_fp80 @llvm.powi.f80.i32(x86_fp80 0xK3FFF8000000000000000, i32 -64) ; <x86_fp80> [#uses=1]
  %conv = fptosi x86_fp80 %tmp to i32             ; <i32> [#uses=1]
  store i32 %conv, i32* %r
  %tmp1 = load i32, i32* %r                            ; <i32> [#uses=1]
  %tobool = icmp ne i32 %tmp1, 0                  ; <i1> [#uses=1]
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @_Z1fv()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %0 = load i32, i32* %retval                          ; <i32> [#uses=1]
  ret i32 %0
}

declare x86_fp80 @llvm.powi.f80.i32(x86_fp80, i32) nounwind readonly

declare void @_Z1fv()
