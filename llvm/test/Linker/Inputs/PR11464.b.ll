%bug_type = type { %bug_type* }
%bar = type { i32 }

define i32 @bug_a(%bug_type* %fp) nounwind uwtable {
entry:
  %d_stream = getelementptr inbounds %bug_type* %fp, i64 0, i32 0
  ret i32 0
}

define i32 @bug_b(%bar* %a) nounwind uwtable {
entry:
  ret i32 0
}
