struct complex_type {
    struct { long l; } inner;
    struct complex_type *complex_ptr;
};

int main() {
    int i = 0;
    struct complex_type c = { { 1L }, &c };
    for (int j = 3; j < 20; j++)
    {
        c.inner.l += (i += j);
        i = i - 1; // break here
    }
    return i;
}
