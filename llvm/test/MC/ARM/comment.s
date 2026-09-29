@ Tests to check that '@' does not get lexed as an identifier for arm
@ RUN: llvm-mc %s -triple=armv7-linux-gnueabi  | FileCheck %s
@ RUN: llvm-mc %s -triple=armv7-linux-gnueabi 2>&1 | FileCheck %s --check-prefix=ERROR

foo:
  bl boo@plt should be ignored
  bl goo@plt
  .long bar@got to parse this as a comment
  .long baz@got
  add r0, r0@ignore this extra junk

@ the symver directive should allow @ in the second symbol name
defined1:
defined2:
defined3:
bar:
  .symver defined1, bar1@zed
  .symver defined2, bar3@@zed
  .symver defined3, bar5@@@zed

far:
  .long baz@got

@CHECK-LABEL: foo:
@CHECK: bl boo
@CHECK-NOT: @
@CHECK: bl goo
@CHECK-NOT: @
@CHECK: .long bar
@CHECK-NOT: @
@CHECK: .long baz
@CHECK-NOT: @
@CHECK: add r0, r0
@CHECK-NOT: @

@CHECK-LABEL: bar:
@CHECK: .symver defined1, bar1@zed
@CHECK: .symver defined2, bar3@@zed
@CHECK: .symver defined3, bar5@@@zed

@ Make sure we did not mess up the parser state and it still lexes
@ comments correctly by excluding the @ in normal symbols
@CHECK-LABEL: far:
@CHECK:  .long baz
@CHECK-NOT: @

@ERROR-NOT: error:
