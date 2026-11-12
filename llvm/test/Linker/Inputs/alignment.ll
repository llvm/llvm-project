@A = global i32 7, align 8
@B = global i32 7, align 4

define void @C() align 8 {
  ret void
}

define void @D() align 4 {
  ret void
}

@E = common global i32 0, align 8
