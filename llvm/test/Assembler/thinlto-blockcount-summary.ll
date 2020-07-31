; Test that we can parse blockcount summaries.
; RUN: opt %s -o /dev/null

define void @main() {
entry:
    ret void
}

^0 = module: (path: "{{.*}}thinlto-bad-summary5.ll", hash: (0, 0, 0, 0, 0))
^1 = blockcount: 1234
