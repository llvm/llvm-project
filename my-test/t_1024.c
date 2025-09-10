int f(int x) {
    int sum = 0;
    for (int i = 0; i <= 1023; i++) {
        sum += (x + i);
    }
    return sum;
}