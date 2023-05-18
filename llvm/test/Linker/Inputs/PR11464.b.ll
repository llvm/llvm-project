%bug_type = type { ptr }
%bar = type { i32 }

define i32 @bug_a(ptr %fp) nounwind uwtable {
entry:
  ret i32 0
}

define i32 @bug_b(ptr %a) nounwind uwtable {
entry:
  ret i32 0
}
