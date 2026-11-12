void *D = 0;

class D {
    static int i;
};

int D::i = 3;

namespace errno {
    int j = 4;
};

int twice(int n)
{
    return n * 2; //% self.expect("expression -- D::i", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["int", "3"])
                  //% self.expect("expression -- D", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["void"])
                  //% self.expect("expression -- errno::j", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["int", "4"])
}

const char getAChar()
{
    const char D[] = "Hello world";
    return D[0];  //% self.expect("expression -- D::i", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["int", "3"])
                  //% self.expect("expression -- D", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["char", "Hello"])
}

int main (int argc, char const *argv[])
{
    int six = twice(3);
    return 0; 
}
