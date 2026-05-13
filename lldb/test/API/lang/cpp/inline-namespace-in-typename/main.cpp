namespace foo {
struct Duplicate {
} t1;

inline namespace bar {
struct Duplicate {
} t2;
struct Unique {
} t3;
} // namespace bar
} // namespace foo

int main() { return 0; }
