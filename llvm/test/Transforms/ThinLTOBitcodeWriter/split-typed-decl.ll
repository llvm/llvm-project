;; Generating bitcode files with split LTO modules should not crash if there are
;; typed declarations in sources.

; RUN: opt --thinlto-bc --thinlto-split-lto-unit -o - %s

@_ZTV3Foo = external constant { [3 x ptr] }, !type !0

define void @Bar() {
  ret void
}

!0 = !{i64 16, !"_ZTS3Foo"}
