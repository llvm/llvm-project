int main(void)
{
    __int128_t n = 1;
    n = n + n;
    return n; //%self.expect("expression n", substrs=['(__int128_t) $0 = 2'])
              //%self.expect("expression n + 6", substrs=['(__int128_t) $1 = 8'])
              //%self.expect("expression n + n", substrs=['(__int128_t) $2 = 4'])
}
