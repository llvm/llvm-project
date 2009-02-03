-- RUN: %llvmgcc -c %s
procedure VCE_LV is
   type P is access String ;
   type T is new P (5 .. 7);
   subtype U is String (5 .. 7);
   X : T := new U'(others => 'A');
begin
   null;
end;
