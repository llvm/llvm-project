[[gnu::weak]] void lib1_internal();

int main() {
    lib1_internal();
    __builtin_debugtrap();
}
