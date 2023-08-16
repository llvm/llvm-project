; RUN: opt -passes='simplifycfg<sink-common-insts;hoist-common-insts>,verify' -disable-output %s

; XFAIL: *

declare void @clobber1()
declare void @clobber2()

; FIXME: currently simplifycfg tries to sink the stores to the exit block and
; introduces a select for the pointer operand. This is not allowed for
; swifterror pointers.
define swiftcc void @sink_store(ptr %arg, ptr swifterror %arg1, i1 %c) {
bb:
  br i1 %c, label %then, label %else

then:
  call void @clobber1()
  store ptr null, ptr %arg, align 8
  br label %exit

else:
  call void @clobber2()
  store ptr null, ptr %arg1, align 8
  br label %exit

exit:
  ret void
}

define swiftcc void @hoist_store(ptr %arg, ptr swifterror %arg1, i1 %c) {
bb:
  br i1 %c, label %then, label %else

then:
  store ptr null, ptr %arg, align 8
  call void @clobber1()
  br label %exit

else:
  store ptr null, ptr %arg1, align 8
  call void @clobber2()
  br label %exit

exit:
  ret void
}

; FIXME: currently simplifycfg tries to sink the load to the exit block and
; introduces a select for the pointer operand. This is not allowed for
; swifterror pointers.
define swiftcc ptr @sink_load(ptr %arg, ptr swifterror %arg1, i1 %c) {
bb:
  br i1 %c, label %then, label %else

then:
  call void @clobber1()
  %l1 = load ptr, ptr %arg, align 8
  br label %exit

else:
  call void @clobber2()
  %l2 = load ptr, ptr %arg1, align 8
  br label %exit

exit:
  %p = phi ptr [ %l1, %then ], [ %l2, %else ]
  ret ptr %p
}
define swiftcc ptr @hoist_load(ptr %arg, ptr swifterror %arg1, i1 %c) {
bb:
  br i1 %c, label %then, label %else

then:
  %l1 = load ptr, ptr %arg, align 8
  call void @clobber1()
  br label %exit

else:
  %l2 = load ptr, ptr %arg1, align 8
  call void @clobber2()
  br label %exit

exit:
  %p = phi ptr [ %l1, %then ], [ %l2, %else ]
  ret ptr %p
}
