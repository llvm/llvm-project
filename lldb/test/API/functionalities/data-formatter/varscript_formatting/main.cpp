template <typename T>
struct something {};

int main() {
    something<int> x;
    something<void*> y;
    return 0; // Set breakpoint here.
}
