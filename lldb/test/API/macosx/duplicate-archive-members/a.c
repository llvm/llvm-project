int __a_global = 1;

int a(int arg) {
    int result = arg + __a_global;
    return result;
}
