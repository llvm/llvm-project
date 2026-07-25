; RUN: opt -passes=sandbox-vectorizer -sbvec-passes="regions-from-metadata<null(aux-arg-foo)>" %s -disable-output

; Checks that the NullPass, which is a region pass can take an aux argument.
; TODO: This test can be removed once real passes start using aux-args.

define void @foo() {
  ret void, !sandboxvec !0
}
!0 = distinct !{!"sandboxregion"}

