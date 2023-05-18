class PrintfContainer {
public:
    int printf() {
        return 0;
    }
};

int main() {
    PrintfContainer().printf(); //% self.expect("expression -- printf(\"Hello\\n\")", substrs = ['6'])
    return 0;
}

