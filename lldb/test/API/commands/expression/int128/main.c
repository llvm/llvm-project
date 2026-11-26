int main(void)
{
    __int128_t n = 1;
    n = n + n;
    return n; //%int128 = "__int128" if self.getDebugInfo() == "pdb" else "__int128_t"
              //%self.expect("expression n", substrs=['(%s) $0 = 2' % int128])
              //%self.expect("expression n + 6", substrs=['(%s) $1 = 8' % int128])
              //%self.expect("expression n + n", substrs=['(%s) $2 = 4' % int128])
}
