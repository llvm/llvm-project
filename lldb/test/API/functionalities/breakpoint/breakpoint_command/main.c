int main (int argc, char const *argv[])
{
    // Add a body to the function, so we can set more than one
    // breakpoint in it.
    static volatile int var = 0;
    var++;
    return 0; // Set break point at this line.
}
