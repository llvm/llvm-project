; RUN: %lli %s > /dev/null

define i64 @foo() {
  ret i64 42
}

define i32 @main() {
  %val = call i64 @foo()
  %is42 = icmp eq i64 %val, 42
  br i1 %is42, label %good, label %bad

good:
  ret i32 0

bad:
  ret i32 1
} 
