let array = [SampleEnum.zero, SampleEnum.one, SampleEnum.two, SampleEnum.three]
print(array) //%self.expect('frame variable array[0]', substrs=['[0]', '.SampleEnumZero'])
             //%self.expect('frame variable array[1]', substrs=['[1]', '.SampleEnumOne'])
             //%self.expect('frame variable array[2]', substrs=['[2]', '.SampleEnumTwo'])
             //%self.expect('frame variable array[3]', substrs=['[3]', '.SampleEnumThree'])
