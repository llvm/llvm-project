int frame3() { return 3; }

int frame2() { return frame3(); }

int frame1() { return frame2(); }

int main() { return frame1(); }
