# RUN: not llvm-mc -triple i386-unknown-unknown %s 2>&1 | FileCheck %s
error
# CHECK: hash-directive.s:[[@LINE-1]]:1: error
# 3 "FILE1" 1 #<- This is a CPP Hash w/ comment
error
# CHECK: FILE1:3:1: error
# 0 "" 2 #<- This is too
error
# CHECK: hash-directive.s:[[@LINE-1]]:1: error
 # 1 "FILE2" 2 #<- This is a comment
error
# CHECK: hash-directive.s:[[@LINE-1]]:1: error
nop; # 6 "FILE3" 2 #<- This is a still comment
error
# CHECK: hash-directive.s:[[@LINE-1]]:1: error
nop;# 6 "FILE4" 2
  nop;
error
# CHECK: FILE4:7:1: error
# 0 "" 2
/*comment*/# 6 "FILE5" 2 #<- This is a comment
error
# CHECK: hash-directive.s:[[@LINE-1]]:1: error
