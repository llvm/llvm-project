; RUN: llvm-as < %s | llc

define i32 @main() {  
  ret i32 42
}
