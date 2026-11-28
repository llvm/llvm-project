; Test that we can parse flag summaries.
; RUN: opt %s -disable-output

define void @main() {
entry:
    ret void
}

^0 = module: (path: "{{.*}}thinlto-flags-summary.ll", hash: (0, 0, 0, 0, 0))
^1 = flags: 8
