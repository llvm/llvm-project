N = 1_000_000  # default: one million temporaries
print("define i64 @massive() {")
print("entry:")
for i in range(N):
    print(f"  %v{i} = add i64 {i}, {i+1}")
sumvar = "%v0"
for i in range(1, N):
    print(f"  %sum{i} = add i64 {sumvar}, %v{i}")
    sumvar = f"%sum{i}"
print(f"  ret i64 {sumvar}")
print("}")
