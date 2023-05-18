int pat (int in) { 
  return in + 5; // break here 
}

int tat (int in) { return pat(in + 10); }

int mat (int in) { return tat(in + 15); }

int main() {
 int (*matp)(int) = mat;
 return matp(10);
}
