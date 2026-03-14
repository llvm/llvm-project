; RUN: llc < %s

; NVPTX failed to lower arg i160, as size > 64
; UNSUPPORTED: target=nvptx{{.*}}

@c = external global i32, align 4

define void @foo(i160 %x) {
entry:
  %cmp.i = icmp ne i160 %x, 340282366920938463463374607431768211456
  %conv.i = zext i1 %cmp.i to i32
  %tobool.i = icmp eq i32 %conv.i, 0
  br i1 %tobool.i, label %if.then.i, label %fn1.exit

if.then.i:
  store i32 0, ptr @c, align 4
  br label %fn1.exit

fn1.exit:
  ret void
}
