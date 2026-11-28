; Check that the plugin's RegisterTargetPassConfigCallback inserts its pass
; REQUIRES: system-linux, plugins, examples

; RUN: %if x86-registered-target %{ llc -mtriple=x86_64 -load-pass-plugin %llvmshlibdir/Ciao%pluginext %s -o /dev/null | FileCheck %s %}
; RUN: %if aarch64-registered-target %{ llc -mtriple=aarch64 -load-pass-plugin %llvmshlibdir/Ciao%pluginext %s -o /dev/null | FileCheck %s %}

; CHECK: Ciao: main
define void @main() {
  ret void
}
