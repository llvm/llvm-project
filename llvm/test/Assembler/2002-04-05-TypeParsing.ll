; RUN: llvm-as %s -o /dev/null -f
  
 %Hosp = type { { \2*, { \2, %Hosp }* }, { \2*, { \2, %Hosp }* } }
