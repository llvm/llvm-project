### If you want to add intrinsic call in your logger, you should do 3 steps:

1) Add signature of intrinsic in global scope. 
``` 
declare i64* @llvm.returnaddress(i32) 
```
2) Add local variable. %3 will have i64* type.
We use 0 as a parameter for returnaddress, because we want to get return address in parent.

```
%3 = call i64* @llvm.returnaddress(i32 0)
``` 
3) Repeat second step, but use 1 as a parameter of returnaddress, because now we need to get return address
of our parent

    
**That's all now you will have a pair of addresses: caller -> callee**
