; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -o /dev/null

declare { {} } @callee()

define i32 @caller() {
entry:
  %call = tail call { {} } @callee()
  ret i32 0
}
