target triple = "thumbv7-linux-gnueabihf"

define i32 @foo(i32 %a, i32 %b) #0 {
entry:
  %add = add i32 %a, %b
  ret i32 %add
}

define i32 @bar(i32 %a, i32 %b) #1 {
entry:
  %add = add i32 %a, %b
  ret i32 %add
}

attributes #0 = { "target-features"="-thumb-mode" }
attributes #1 = { "target-features"="+thumb-mode" }
