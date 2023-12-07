; RUN: llc -mtriple arm-eabi -mcpu swift -verify-machineinstrs %s -o /dev/null

declare i32 @f(i32 %p0, i32 %p1)

define i32 @foo(ptr %ptr) {
entry:
  %cmp = icmp ne ptr %ptr, null
  br i1 %cmp, label %if.then, label %if.else

; present something which can be easily if-converted
if.then:
  ; %R0 should be killed here
  %valt = load i32, ptr %ptr, align 4
  br label %return

if.else:
  ; %R0 should be killed here, however after if-conversion the %R0 kill
  ; has to be removed because if.then will follow after this and still
  ; read it.
  %addr = getelementptr inbounds i32, ptr %ptr, i32 4
  %vale = load i32, ptr %addr, align 4
  br label %return

return:
  %phival = phi i32 [ %valt, %if.then ], [ %vale, %if.else ]
  ; suggest to bring %phival/%valt/%vale into %R1 (because otherwise there
  ; will be no kills in if.then/if.else)
  %retval = call i32 @f (i32 0, i32 %phival)
  ret i32 %retval
}
