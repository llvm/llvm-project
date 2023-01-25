struct MyStruct {
  int before;
  char var[5];
  int after;
};

int
main()
{
  struct MyStruct struct_arr[3] = {{112, "abcd", 221},
                                   {313, "efgh", 414},
                                   {515, "ijkl", 616}};
  struct MyStruct *struct_ptr = struct_arr;
  return struct_ptr->before;  // Set a breakpoint here
}
