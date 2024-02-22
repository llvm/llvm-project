struct SomeTypeWeWillLookUp {
}

struct SomeNewTypeToBreakTheCache {
}

let v = SomeTypeWeWillLookUp()
print(v) // Set breakpoint here
