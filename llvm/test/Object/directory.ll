;RUN: rm -rf %t && mkdir -p %t
;RUN: not llvm-ar r %t/test.a . 2>&1 | FileCheck -DMSG=%errc_EISDIR %s
;CHECK: .: [[MSG]]

;RUN: rm -f %t/test.a
;RUN: touch %t/a-very-long-file-name
;RUN: llvm-ar r %t/test.a %s %t/a-very-long-file-name
;RUN: llvm-ar r %t/test.a %t/a-very-long-file-name
;RUN: llvm-ar t %t/test.a | FileCheck -check-prefix=MEMBERS %s
;MEMBERS-NOT: /
;MEMBERS: directory.ll
;MEMBERS: a-very-long-file-name
;MEMBERS-NOT: a-very-long-file-name
