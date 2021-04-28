; REQUIRES: plugins
; RUN: echo "import sys" > %t.py
; RUN: echo "print('args = ' + str(sys.argv))" >> %t.py
; RUN: echo "exit(1)" >> %t.py
; RUN: not bugpoint -load %llvmshlibdir/BugpointPasses%pluginext %s -output-prefix %t -bugpoint-crashcalls -opt-command=%python -opt-args %t.py | FileCheck %s
; RUN: not --crash opt -enable-new-pm=0 -load %llvmshlibdir/BugpointPasses%pluginext %s -bugpoint-crashcalls -disable-symbolication 2>&1 | FileCheck --check-prefix=CRASH %s
; RUN: not bugpoint -load %llvmshlibdir/BugpointPasses%pluginext %s -output-prefix %t -bugpoint-crashcalls -opt-command=%t.non.existent.opt.binary -opt-args %t.py 2>&1 | FileCheck %s --check-prefix=BAD-OPT

; Test that bugpoint disables symbolication on the opt tool to reduce runtime overhead when opt crashes
; CHECK: args = {{.*}}'-disable-symbolication'

; Test that opt, when it crashes & is passed -disable-symbolication, doesn't symbolicate.
; In theory this test should maybe be in test/tools/opt or
; test/Transforms, but since there doesn't seem to be another convenient way to
; crash opt, apart from the BugpointPasses dynamic plugin, this is the spot for
; now.
; CRASH-NOT: Signals.inc

; BAD-OPT: Specified `opt' binary does not exist: {{.*}}non.existent.opt.binary
define void @f() {
  call void @f()
  ret void
}
