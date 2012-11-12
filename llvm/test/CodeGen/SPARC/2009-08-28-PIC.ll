; RUN: llc -march=sparc --relocation-model=pic < %s | grep _GLOBAL_OFFSET_TABLE_

@foo = global i32 0                               ; <i32*> [#uses=1]

define i32 @func() nounwind readonly {
entry:
  %0 = load i32* @foo, align 4                    ; <i32> [#uses=1]
  ret i32 %0
}
