@import ObjectiveC;
@import myModule;

int main() {
    MyClass *m = [[MyClass alloc] init];
    int i = m.propConflict + MyClass.propConflict;
    return i; // Set breakpoint here.
}
