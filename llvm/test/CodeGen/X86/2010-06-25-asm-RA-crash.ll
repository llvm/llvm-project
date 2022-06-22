; RUN: llc < %s -frame-pointer=all -mtriple=i686-pc-mingw32 -no-integrated-as

%struct.__SEH2Frame = type {}

define void @_SEH2FrameHandler() nounwind {
entry:
  %target.addr.i = alloca ptr, align 4            ; <ptr> [#uses=2]
  %frame = alloca ptr, align 4   ; <ptr> [#uses=1]
  %tmp = load ptr, ptr %frame        ; <ptr> [#uses=1]
  store ptr %tmp, ptr %target.addr.i
  %tmp.i = load ptr, ptr %target.addr.i               ; <ptr> [#uses=1]
  call void asm sideeffect "push %ebp\0Apush $$0\0Apush $$0\0Apush $$Return${:uid}\0Apush $0\0Acall ${1:c}\0AReturn${:uid}: pop %ebp\0A", "imr,imr,~{ax},~{bx},~{cx},~{dx},~{si},~{di},~{flags},~{memory},~{dirflag},~{fpsr},~{flags}"(ptr %tmp.i, ptr @RtlUnwind) nounwind, !srcloc !0
  ret void
}

declare x86_stdcallcc void @RtlUnwind(...)

!0 = !{i32 215}
